"""Manager for decoder worker process.

Spawns, monitors, and restarts the decoder worker process if it crashes.
"""

import asyncio
import logging
import multiprocessing as mp
import queue
import time
from typing import Any, Dict, Optional, Tuple

_logger: logging.Logger = logging.getLogger(__name__)


class DecoderWorkerManager:
    """Manages a persistent decoder worker process."""
    
    def __init__(self, maxsize: int = 100):
        """Initialize the decoder worker manager.
        
        Args:
            maxsize: Maximum size for output frame queue (NAL queue will be larger)
        """
        self.maxsize: int = maxsize
        self.nal_queue_size: int = max(200, maxsize * 4)  # NAL queue should be much larger than frame queue
        self._process: Optional[mp.Process] = None
        self._nal_queue: Optional[mp.Queue[Tuple[bytes, Optional[float]]]] = None
        self._frame_queue: Optional[mp.Queue[Tuple[Dict[str, Any], Optional[float]]]] = None
        self._control_queue: Optional[mp.Queue[str]] = None
        self._log_queue: Optional[mp.Queue[Tuple[str, str]]] = None
        
        # Statistics
        self._spawned_count: int = 0
        self._restarted_count: int = 0
        self._nals_submitted: int = 0
        self._frames_received: int = 0
        self._last_restart_time: float = 0.0
        
        # Monitoring
        self._log_consumer_task: Optional[asyncio.Task[None]] = None
    
    def _spawn_worker(self) -> None:
        """Spawn a new decoder worker process."""
        # Create fresh queues
        # NAL queue much larger than frame queue to avoid bottlenecks
        self._nal_queue = mp.Queue(maxsize=self.nal_queue_size)  # type: ignore
        self._frame_queue = mp.Queue(maxsize=self.maxsize)  # type: ignore
        self._control_queue = mp.Queue(maxsize=10)  # type: ignore
        self._log_queue = mp.Queue(maxsize=100)  # type: ignore
        
        # Import here to avoid circular dependencies
        from g3pylib._decoder_worker import decoder_worker # type: ignore
        
        # Spawn process
        self._process = mp.Process(
            target=decoder_worker, # type: ignore
            args=(self._nal_queue, self._frame_queue, self._control_queue, self._log_queue),
            daemon=True
        )
        self._process.start()
        
        self._spawned_count += 1
        self._last_restart_time = time.time()
        
        _logger.info(f"Spawned decoder worker (PID {self._process.pid}, spawn #{self._spawned_count})")
    
    def _restart_worker(self) -> None:
        """Restart the worker process after a crash."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        
        self._restarted_count += 1
        _logger.warning(f"Restarting decoder worker (restart #{self._restarted_count})")
        
        self._spawn_worker()
    
    def _check_worker_alive(self) -> bool:
        """Check if worker is alive, restart if needed."""
        if self._process is None:
            return False
        
        if not self._process.is_alive():
            exitcode = self._process.exitcode
            _logger.error(f"Decoder worker died (exit code {exitcode})")
            self._restart_worker()
            return False
        
        return True
    
    async def start(self) -> None:
        """Start the decoder worker and log consumer."""
        self._spawn_worker()
        
        # Start log consumer task
        self._log_consumer_task = asyncio.create_task(self._consume_logs())
    
    async def _consume_logs(self) -> None:
        """Consume log messages from worker."""
        while True:
            try:
                await asyncio.sleep(0.1)
                
                if self._log_queue is None:
                    continue
                
                # Drain log queue
                while True:
                    try:
                        level, message = self._log_queue.get_nowait()
                        log_func: Any = getattr(_logger, level.lower(), _logger.info)
                        log_func(f"[Worker] {message}")
                    except queue.Empty:
                        break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                _logger.error(f"Error consuming logs: {e}")
    
    async def submit_nal(self, nal_data: bytes, timestamp: Optional[float]) -> bool:
        """Submit a NAL unit for decoding.
        
        Args:
            nal_data: NAL unit data with start code prefix
            timestamp: Optional NTP timestamp
            
        Returns:
            True if submitted successfully, False if worker died
        """
        if not self._check_worker_alive():
            return False

        nal_queue = self._nal_queue
        if nal_queue is None:
            _logger.error("NAL queue not initialized; call start() before submit_nal()")
            return False
        
        try:
            # Non-blocking put with longer timeout (500ms instead of 100ms)
            nal_queue.put((nal_data, timestamp), timeout=0.5)
            self._nals_submitted += 1
            return True
        except queue.Full:
            _logger.warning(f"NAL queue full (size: {nal_queue.qsize()}), dropping NAL")
            return False
        except Exception as e:
            _logger.error(f"Error submitting NAL: {e}")
            return False
    
    async def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[Dict[str, Any], Optional[float]]]:
        """Get a decoded frame from the worker.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            (frame_data, timestamp) or None if no frame available
        """
        if not self._check_worker_alive():
            return None
        if self._frame_queue is None:
            _logger.error("Frame queue not initialized; call start() before get_frame()")
            return None
        frame_queue = self._frame_queue
        
        try:
            # Run blocking get in executor to not block event loop
            loop = asyncio.get_event_loop()
            frame_data, timestamp = await loop.run_in_executor(
                None,
                lambda: frame_queue.get(timeout=timeout)
            )
            self._frames_received += 1
            return (frame_data, timestamp)
        except queue.Empty:
            return None
        except Exception as e:
            _logger.error(f"Error getting frame: {e}")
            return None
    
    def frame_queue_size(self) -> int:
        """Get approximate size of frame queue."""
        if self._frame_queue is None:
            return 0
        return self._frame_queue.qsize()
    
    async def stop(self) -> None:
        """Stop the decoder worker gracefully."""
        # Stop log consumer
        if self._log_consumer_task:
            self._log_consumer_task.cancel()
            try:
                await self._log_consumer_task
            except asyncio.CancelledError:
                pass
        
        # Send stop signal
        if self._control_queue:
            try:
                self._control_queue.put("stop", timeout=0.5)
            except Exception:
                pass
        
        # Wait for process to exit
        if self._process and self._process.is_alive():
            self._process.join(timeout=2.0)
            
            # Force kill if still alive
            if self._process.is_alive():
                _logger.warning("Decoder worker didn't stop gracefully, killing")
                self._process.kill()
                self._process.join()
        
        _logger.info("Decoder worker manager stopped")
    
    def stats(self) -> Dict[str, int]:
        """Get worker statistics."""
        return {
            'spawned': self._spawned_count,
            'restarted': self._restarted_count,
            'nals_submitted': self._nals_submitted,
            'frames_received': self._frames_received,
            'alive': self._process.is_alive() if self._process else False,
            'frame_queue_size': self.frame_queue_size(),
        }

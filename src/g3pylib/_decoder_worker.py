"""Decoder worker process that runs PyAV decode in isolation.

This worker runs the entire NAL parsing and decoding in a separate process.
If it crashes (0xC0000005 or any other error), the parent can restart it.
"""

import logging
import multiprocessing as mp
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

import av  # type: ignore

_logger: logging.Logger = logging.getLogger(__name__)


def decoder_worker(
    nal_queue: "mp.Queue[Tuple[bytes, Optional[float]]]",
    frame_queue: "mp.Queue[Tuple[Dict[str, Any], Optional[float]]]",
    control_queue: "mp.Queue[str]",
    log_queue: "mp.Queue[Tuple[str, str]]"
) -> None:
    """Worker process that decodes NAL units.
    
    Args:
        nal_queue: Input queue receiving (nal_data: bytes, timestamp: Optional[float])
        frame_queue: Output queue sending (frame_data: dict, timestamp: Optional[float])
        control_queue: Control signals ('stop', 'ping', etc.)
        log_queue: Logging messages from worker
    """
    try:
        # log_queue is accepted for API compatibility but logging is handled locally
        _ = log_queue

        _logger.info(f"Decoder worker started (PID {mp.current_process().pid})")
        
        # Create codec context
        codec_context: Any = av.CodecContext.create("h264", "r")  # type: ignore
        _logger.info("Codec context created")
        
        decode_count: int = 0
        parse_count: int = 0
        
        while True:
            # Check control queue for stop signal
            if not control_queue.empty():
                cmd: str = control_queue.get_nowait()
                if cmd == "stop":
                    _logger.info("Stop signal received")
                    break
                elif cmd == "ping":
                    control_queue.put("pong")
            
            # Get NAL from queue (non-blocking with timeout)
            try:
                nal_data: bytes
                timestamp: Optional[float]
                nal_data, timestamp = nal_queue.get(timeout=0.1)
            except Exception:
                continue
            
            try:
                # Parse NAL
                packets: Any = codec_context.parse(nal_data)
                parse_count += 1
                
                if not packets:
                    continue
                
                # Decode packets
                for packet in packets:
                    frames: Any = codec_context.decode(packet)
                    for frame in frames:
                        decode_count += 1
                        
                        # Convert frame to transferable format
                        # We can't send av.VideoFrame directly, so convert to dict
                        frame_data: Dict[str, Any] = {
                            'width': frame.width,
                            'height': frame.height,
                            'format': str(frame.format),
                            'pts': frame.pts,
                            'time_base_num': frame.time_base.numerator if frame.time_base else 1,
                            'time_base_den': frame.time_base.denominator if frame.time_base else 1,
                            # Convert to numpy array then bytes for transfer
                            'data': frame.to_ndarray(format='bgr24').tobytes(),
                        }
                        
                        frame_queue.put((frame_data, timestamp))
                        
                        if decode_count % 100 == 0:
                            _logger.debug(f"Decoded {decode_count} frames, parsed {parse_count} NALs")
                
            except av.InvalidDataError as e:
                # Invalid data errors are common after restart when codec context is fresh
                # Skip this NAL and continue - next valid SPS/PPS will reinitialize the context
                _logger.debug(f"Skipping invalid NAL data (will resume on next SPS/PPS): {str(e)}")
                continue
            except Exception as e:
                # Log other parse/decode errors but continue
                _logger.warning(f"Decode error: {type(e).__name__}: {str(e)}")
                continue
        
        _logger.info(f"Worker stopping normally (decoded {decode_count} frames)")
        sys.exit(0)
        
    except Exception as e:
        # Fatal error in worker
        _logger.error(f"Worker crashed: {type(e).__name__}: {str(e)}")
        _logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # This allows the worker to be spawned correctly on Windows
    mp.freeze_support()

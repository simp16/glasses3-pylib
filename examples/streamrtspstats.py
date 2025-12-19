"""Test script for safe_decode mode with decoder worker process.

This uses the full process isolation approach where entire decode runs
in a separate process that can be restarted on crashes.
"""

import asyncio
import logging
import os
import sys

import cv2
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import g3pylib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Test safe_decode mode with live streaming."""
    load_dotenv()
    
    hostname = os.getenv("G3_HOSTNAME")
    if not hostname:
        logger.error("G3_HOSTNAME not set in .env file")
        return
    
    rtsp_url = f"rtsp://{hostname}:8554/live/all"
    logger.info(f"Connecting to {rtsp_url} (safe_decode defaults to True)")
    
    frame_count = 0
    
    try:
        async with g3pylib.Streams.connect(
            rtsp_url,
            scene_camera=True,
        ) as streams:
            
            async with streams.scene_camera.decode() as frame_queue:
                await streams.play()
                logger.info("Streaming started with safe_decode mode")
                logger.info("Press 'q' in video window or Ctrl+C to quit")
                
                while True:
                    frame, timestamp = await frame_queue.get()
                    frame_count += 1
                    
                    # Convert to numpy for display
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Get stats every 10 frames
                    stats = streams.scene_camera.stats
                    
                    # Extract stats
                    crashed = stats.get('crashed_count', 0)
                    spawned = stats.get('decoder_spawned', 0)
                    restarted = stats.get('decoder_restarted', 0)
                    nals_submitted = stats.get('decoder_nals_submitted', 0)
                    frames_received = stats.get('decoder_frames_received', 0)
                    alive = stats.get('decoder_alive', False)
                    
                    # Add text overlay
                    cv2.putText(img, f"Frames: {frame_count} | Crashed: {crashed}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Decoder: Spawned={spawned} Restarted={restarted} Alive={alive}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"NALs Submitted: {nals_submitted} | Frames Recv: {frames_received}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                    # Display frame
                    cv2.imshow("Scene Camera (Safe Decode - Process Isolation)", img)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("'q' pressed, exiting")
                        break
                    
                    # Log stats every 100 frames
                    if frame_count % 1000 == 0:
                        stats = streams.scene_camera.stats
                        logger.info(f"Frame {frame_count}: {stats}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()
        logger.info(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    asyncio.run(main())

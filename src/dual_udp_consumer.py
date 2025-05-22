# market_data_consumer/app.py
import socket
import json
import os
import time
import logging
import asyncio

# Protobuf imports (adjust path)
# Assuming market_data_pb2.py will be in market_data_consumer/protos/
from protos import market_data_pb2

# NATS client
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers

# Prometheus client
from prometheus_client import Counter, Gauge, start_http_server

# --- Configuration ---
LISTEN_IP = os.getenv('LISTEN_IP', '0.0.0.0')
LISTEN_PORT = int(os.getenv('LISTEN_PORT', 9999))
BUFFER_SIZE = 2048
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 8000))
NATS_URL = os.getenv('NATS_URL', 'nats://localhost:4222')
NATS_PROD_TYPE_PREFIX = os.getenv('NATS_PROD_TYPE_PREFIX', 'prodtype')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# --- Logging Setup ---
# (Keep your existing logging setup)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --- Prometheus Metrics ---
# (Keep your existing Prometheus metrics definitions)
RECEIVED_PACKETS = Counter('marketdata_received_packets_total', 'Total received UDP packets', ['type'])
PROCESSED_MESSAGES = Counter('marketdata_processed_messages_total', 'Total processed messages sent to NATS', ['type', 'topic'])
PROCESSING_ERRORS = Counter('marketdata_processing_errors_total', 'Total errors during processing', ['error_type'])
NATS_PUBLISH_ERRORS = Counter('marketdata_nats_publish_errors_total', 'Total errors publishing to NATS')


# --- NATS Connection ---
nc = NATS()
nats_connected = False

async def nats_connect():
    global nc, nats_connected
    if not nats_connected or nc.is_closed:
        try:
            logger.info(f"Attempting to connect to NATS at {NATS_URL}")
            await nc.connect(servers=[NATS_URL],
                             reconnect_time_wait=5,
                             max_reconnect_attempts=60)
            nats_connected = True
            logger.info(f"Connected to NATS at {NATS_URL}")
        except ErrNoServers:
            logger.error(f"Could not connect to NATS: No servers available at {NATS_URL}. Retrying...")
            nats_connected = False
        except Exception as e:
            logger.error(f"Error connecting to NATS: {e}. Retrying...")
            nats_connected = False
    return nats_connected

async def publish_to_nats(topic: str, data: bytes):
    global nc
    if not await nats_connect():
        logger.warning(f"NATS not connected. Cannot publish message to {topic}")
        NATS_PUBLISH_ERRORS.inc()
        return False

    try:
        await nc.publish(topic, data)
        logger.debug(f"Published message to NATS topic: {topic}, size: {len(data)}")
        PROCESSED_MESSAGES.inc({'type': topic.split('+')[-1], 'topic': topic})
        return True
    except ErrConnectionClosed:
        logger.error("NATS connection closed. Attempting to reconnect.")
        global nats_connected # pylint: disable=global-statement
        nats_connected = False
        NATS_PUBLISH_ERRORS.inc()
    except ErrTimeout:
        logger.error("Timeout publishing to NATS. Message may be lost.")
        NATS_PUBLISH_ERRORS.inc()
    except Exception as e:
        logger.error(f"Error publishing to NATS: {e}")
        NATS_PUBLISH_ERRORS.inc()
    return False


async def main_loop():
    # (Keep the main_loop logic exactly as before, just ensure it's an async function)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        client_socket.bind((LISTEN_IP, LISTEN_PORT))
        logger.info(f"UDP Listener started on {LISTEN_IP}:{LISTEN_PORT}")
    except OSError as e:
        logger.critical(f"Failed to bind UDP socket to {LISTEN_IP}:{LISTEN_PORT}: {e}")
        PROCESSING_ERRORS.inc({'error_type': 'socket_bind_error'})
        return

    await nats_connect()

    try:
        while True:
            try:
                client_socket.settimeout(0.1)
                raw_data, addr = client_socket.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                await asyncio.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"Socket recvfrom error: {e}")
                PROCESSING_ERRORS.inc({'error_type': 'socket_recv_error'})
                await asyncio.sleep(1)
                continue

            # Increment before specific type is known
            RECEIVED_PACKETS.labels(type='unknown_initial').inc()


            try:
                message_str = raw_data.decode('utf-8')
                data = json.loads(message_str)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.warning(f"Error decoding/parsing JSON from {addr}: {e}, Data: {raw_data[:60]}...")
                PROCESSING_ERRORS.inc({'error_type': 'json_decode_error'})
                RECEIVED_PACKETS.labels(type='malformed_json').inc()
                continue

            timestamp = data.get("timestamp")
            packet_type = data.get("packet_type")

            if not timestamp or not packet_type:
                logger.warning(f"Received malformed packet (missing timestamp or packet_type): {data}")
                PROCESSING_ERRORS.inc({'error_type': 'missing_fields'})
                RECEIVED_PACKETS.labels(type='missing_fields').inc()
                continue

            proto_message = None
            nats_topic_suffix = ""

            if packet_type == "quote":
                RECEIVED_PACKETS.labels(type='quote_json').inc() # Overwrite 'unknown_initial'
                try:
                    quote_pb = market_data_pb2.QuoteData(
                        timestamp=str(timestamp),
                        symbol=data.get("symbol", ""),
                        bid_price=float(data.get("bid_price", 0.0)),
                        ask_price=float(data.get("ask_price", 0.0)),
                        bid_size=int(data.get("bid_size", 0)),
                        ask_size=int(data.get("ask_size", 0))
                    )
                    proto_message = quote_pb
                    nats_topic_suffix = "quote"
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting quote JSON to Protobuf: {e}, Data: {data}")
                    PROCESSING_ERRORS.inc({'error_type': 'quote_conversion_error'})
                    continue

            elif packet_type == "trade":
                RECEIVED_PACKETS.labels(type='trade_json').inc() # Overwrite 'unknown_initial'
                try:
                    trade_pb = market_data_pb2.TradeData(
                        timestamp=str(timestamp),
                        symbol=data.get("symbol", ""),
                        price=float(data.get("price", 0.0)),
                        volume=int(data.get("volume", 0)),
                        condition=data.get("condition", "")
                    )
                    proto_message = trade_pb
                    nats_topic_suffix = "trade"
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting trade JSON to Protobuf: {e}, Data: {data}")
                    PROCESSING_ERRORS.inc({'error_type': 'trade_conversion_error'})
                    continue
            else:
                logger.warning(f"Unknown packet type for {timestamp}: {packet_type}")
                PROCESSING_ERRORS.inc({'error_type': 'unknown_packet_type'})
                RECEIVED_PACKETS.labels(type='unknown_packet_type').inc() # Overwrite
                continue

            if proto_message:
                try:
                    serialized_data = proto_message.SerializeToString()
                    topic = f"{NATS_PROD_TYPE_PREFIX}+{nats_topic_suffix}"
                    await publish_to_nats(topic, serialized_data)
                except Exception as e:
                    logger.error(f"Error serializing or preparing NATS publish for {packet_type}: {e}")
                    PROCESSING_ERRORS.inc({'error_type': f'{packet_type}_serialization_error'})

    except KeyboardInterrupt:
        logger.info("UDP Listener shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in main_loop: {e}", exc_info=True)
        PROCESSING_ERRORS.inc({'error_type': 'unhandled_exception'})
    finally:
        logger.info("Closing UDP socket.")
        client_socket.close()
        if nc and not nc.is_closed:
            logger.info("Draining and closing NATS connection.")
            await nc.drain()
        logger.info("Shutdown complete.")


def run_main(): # Wrapper for poetry script
    # Start Prometheus metrics server
    try:
        start_http_server(PROMETHEUS_PORT)
        logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
    except OSError as e:
        logger.critical(f"Failed to start Prometheus server on port {PROMETHEUS_PORT}: {e}. Metrics will not be available.")
        PROCESSING_ERRORS.inc({'error_type': 'prometheus_start_error'})

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_loop())
    except KeyboardInterrupt:
        logger.info("Application interrupted. Initiating shutdown sequence.")
    finally:
        if nc and not nc.is_closed: # Ensure NATS is closed if main_loop didn't
            logger.info("Ensuring NATS connection is closed post loop.")
            # Check if loop is still running to run async close
            if loop.is_running():
                loop.run_until_complete(nc.close())
            else: # Fallback to a new loop for closing if the main one is gone
                asyncio.run(nc.close())

        # Only close the loop if we own it and it's not already closed
        if not loop.is_closed():
             loop.close()
        logger.info("Event loop closed.")

if __name__ == "__main__":
    run_main()
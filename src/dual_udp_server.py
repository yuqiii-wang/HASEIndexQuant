import socket
import json
import time
import datetime
import random
import os

# Configuration
CLIENT_HOST = os.getenv('UDP_CLIENT_SERVICE_HOST', '127.0.0.1') # K8s service name or IP
CLIENT_PORT = int(os.getenv('UDP_CLIENT_SERVICE_PORT', 9999)) # K8s service port
SERVER_SEND_INTERVAL_MS = int(os.getenv('SERVER_SEND_INTERVAL_MS', 0)) # 0 for as fast as possible

def generate_timestamp(dt_object):
    return dt_object.strftime('%Y%m%d%H%M%S%f')[:-3]

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP Server starting, will send to {CLIENT_HOST}:{CLIENT_PORT}")

    current_time = datetime.datetime.now()
    base_ask = 12345.0
    base_bid = 12344.0
    base_last_trade = 12344.5

    try:
        while True:
            ts_str = generate_timestamp(current_time)

            # Packet 1: Quote data
            quote_data = {
                "timestamp": ts_str,
                "ask": round(base_ask + random.uniform(-0.5, 0.5) * 5, 2),
                "bid": round(base_bid + random.uniform(-0.5, 0.5) * 5, 2),
                "code": "HSI",
                "packet_type": "quote" # To help client identify
            }

            # Packet 2: Trade data
            trade_data = {
                "timestamp": ts_str,
                "lastTrade": round(base_last_trade + random.uniform(-0.5, 0.5) * 2, 2),
                "lastTradeVolume": random.randint(1, 100),
                "code": "HSI",
                "packet_type": "trade" # To help client identify
            }

            # Send both packets for the same timestamp
            server_socket.sendto(json.dumps(quote_data).encode('utf-8'), (CLIENT_HOST, CLIENT_PORT))
            # print(f"Sent quote: {quote_data}")
            server_socket.sendto(json.dumps(trade_data).encode('utf-8'), (CLIENT_HOST, CLIENT_PORT))
            # print(f"Sent trade: {trade_data}")
            
            # Increment time by 1 ms for the next pair
            current_time += datetime.timedelta(milliseconds=1)

            # Optional: slight variation for next iteration's base prices
            base_ask += random.uniform(-0.1, 0.1)
            base_bid += random.uniform(-0.1, 0.1)
            base_last_trade = (base_ask + base_bid) / 2 + random.uniform(-0.05, 0.05)

            if SERVER_SEND_INTERVAL_MS > 0:
                time.sleep(SERVER_SEND_INTERVAL_MS / 1000.0)
            # For "as fast as possible", no explicit sleep or very small one
            # A very small sleep can prevent 100% CPU usage on one core if Python loop is too tight
            # time.sleep(0.00001) 

    except KeyboardInterrupt:
        print("Server shutting down.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()
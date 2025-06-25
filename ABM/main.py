from visualization import create_server

if __name__ == "__main__":
    server = create_server()
    server.port = 8521
    server.launch()
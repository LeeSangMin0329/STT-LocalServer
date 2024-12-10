import requests

server_url = "http://127.0.0.1:50003"

def test_request_play(message: str):
    if message.strip() == "":
        print("is empty string")
        return 

    try:
        response = requests.post(f"{server_url}/chat", data={"message": message})

        print(f"Send to {server_url} : {response.status_code}")
    except Exception as e:
        print(f"Exception : {str(e)}")

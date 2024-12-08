import requests

server_url = "http://127.0.0.1:50001"

def test_request_play(message: str):
    if message.strip() == "":
        print("is empty string")
        return 

    response = requests.post(f"{server_url}/requset_message_form_outside", data={"message": message})

    print(f"Send to {server_url} : {response.status_code}")

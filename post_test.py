import requests
import base64
def post(image_bytes):

    url = "http://localhost:20011/search"

    data = {
        "image": base64.b64encode(image_bytes).decode("utf-8")
    }
    resp = requests.post(url, json=data)
    data = resp.json()
    image_bytes = base64.b64decode(data["image"])
    return image_bytes


if __name__ == "__main__":
    with open("1.png", "rb") as f:
        image_bytes = f.read()
    with open("show.png", "wb") as f:
        f.write(post(image_bytes))


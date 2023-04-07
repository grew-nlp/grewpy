#===========================================================================
# used for launching grew web on a given corpus
import tempfile
import requests
import webbrowser
import http
import json
from .grew import GrewError


local = False
if local:
    grew_web_back = "http://localhost:8080"
    grew_web_front = "http://localhost:8888/grew_web"
else:
    grew_web_back = "https://gwb.grew.fr"
    grew_web_front = "https://web.grew.fr"


def _post_request(service, resp):
    if resp.status_code >= 300:
        raise GrewError({
            "Error": "HTTP",
            "service": service,
            "status_code": resp.status_code,
            "message": http.client.responses[resp.status_code]
        })
    data = json.loads(resp.text)
    if data["status"] == "OK":
        return data["data"]
    if data["status"] == "ERROR":
        raise GrewError({"grew_web": service, "message": data["message"]})
    raise GrewError({"UNEXPECTED": service, "message": data})


class Grew_web:
    """
    web-connection to grew
    """
    def __init__(self):
        self.session_id = _post_request(
            "connect", requests.post(f"{grew_web_back}/connect"))

    def load_grs(self,grs):
        json_grs = grs.json()
        if self.session_id == "":
            raise GrewError({"grew_web": "not connected"})
        with tempfile.NamedTemporaryFile(mode="a+", delete=True, suffix=".grs") as f:
            f.write(json.dumps(json_grs))
            f.seek(0)  # ready to be read
            r = requests.post(f"{grew_web_back}/upload_grs",
                          data={"session_id": self.session_id},
                          files={"json_file": f}
                          )
            _post_request("upload_grs", r)


    def load_corpus(self,   corpus):
        conll = corpus.to_conll()
        if self.session_id == "":
            raise GrewError({"grew_web": "not connected"})
        with tempfile.NamedTemporaryFile(mode="a+", delete=True, suffix=".conllu") as f:
            f.write(conll)
            f.seek(0)  # ready to be read
            r = requests.post(f"{grew_web_back}/upload_corpus",
                          data={"session_id": self.session_id},
                          files={"file": f}
                          )
            _post_request("upload_corpus", r)


    def url(self):
        if self.session_id == "":
            raise GrewError({"grew_web": "not connected"})
        return f"{grew_web_front}?session_id={self.session_id}"

    def open(self):
        webbrowser.open (self.url())

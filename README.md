# flask_app_fraud_detection
This module deploys fraud detection model (autoencoder) as an API on Heroku

## Usage
To use, create a test.py file which contains transaction details.

An example is given by

```python
import requests

r = requests.post('https://courage-fraud-detection.herokuapp.com/is-fraud', json = {
        "step":1,
        "type":"PAYMENT",
        "amount":9839.64,
        "nameOrig":"C1231006815",
        "oldbalanceOrig":170136.0,
        "newbalanceOrig":160296.36,
        "nameDest":"M1979787155",
        "oldbalanceDest":0.0,
        "newbalanceDest":0.0
    })

print(r.text)
```
Then run on command line

```
$ python test.py
```

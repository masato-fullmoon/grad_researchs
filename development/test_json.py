import pandas as pd
import json

if __name__ == '__main__':
    base_dict = {
            'height':56,
            'width':56,
            'channel':3
            }

    with open('test.json', 'w') as f:
        json.dump(base_dict, f)

    with open('test.json', 'r') as g:
        nanika = json.load(g)
        print(nanika)
        print(type(nanika))

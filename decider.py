import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
import json
import os
from time import time
import random


class dqn(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.model = Sequential()
        self.model.add(Dense(output_dim=1000, input_dim=9901, activation='relu', init='uniform'))
        self.model.add(Dense(output_dim=1000, activation='sigmoid', init='uniform'))
        self.model.add(Dense(output_dim=500, activation='sigmoid', init='uniform'))
        self.model.add(Dense(output_dim=68, activation='sigmoid', init='uniform'))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def get_response(self, stats):
        stats = np.array([stats])
        return self.model.predict(stats)

    def update_weights(self, epochs):
        files = os.listdir('replay_experience/')
        all_data = []
        if len(files) > 800:
            files = files[800:]
        if len(files) > 20:
            files = random.choices(files, k=20)
        for file in files:
            print('Getting ' + file)
            f = open('replay_experience/' + file, 'r')
            try:
                all_data.extend(json.load(f))
                f.close()
            except json.decoder.JSONDecodeError:
                try:
                    f.close()
                    os.remove('replay_experience/' + file)
                    print(file + ' is not included to train data and deleted.')
                except PermissionError:
                    print(file + ' is not included to train data but couldn\'t be deleted.')
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            random.shuffle(all_data)
            data_size = len(all_data)
            data = 0
            td = 0
            for record in all_data:
                data += 1
                x = np.array([record['Status']])
                action = np.array(record['Action'])
                q = record['Q']
                y = self.model.predict(x)
                td = td*0.99 + 0.01 * abs(q-y[0][int(action[0])])
                if 6 < action[0] < 27:
                    xind, yind = 2 * action[0] + 14, 2 * action[0] + 15
                    y[0][int(xind)], y[0][int(yind)] = action[1], action[2]
                y[0][int(action[0])] = q
                print(f'Epoch : {epoch + 1}/{epochs}\tData : {data}/{data_size}\tTD : {td:.5}')
                self.model.fit(x, y, verbose=False)
        self.model.save(f'models/{time()}.h5')

    def reload(self):
        try:
            files = os.listdir('models/')
            file = open('models/' + files[-1], 'r')
            self.model = load_model(file)
        except:
            pass

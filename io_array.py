import numpy as np
import random
from PIL import Image

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_game_status(player, players):
    all_stats = np.zeros((32,10))
    all_stats[1][7] = player.hp/100
    all_stats[2][7] = player.gold/50
    all_stats[3][7] = player.xp/200
    lenght = len(player.card)
    if lenght>0:
        all_stats[1][8] = (5*player.card[0].stats['champ_id'] + 1)/300
    if lenght>1:
        all_stats[1][9] = (5*player.card[1].stats['champ_id'] + 1)/300
    if lenght>2:
        all_stats[2][8] = (5*player.card[2].stats['champ_id'] + 1)/300
    if lenght>3:
        all_stats[2][9] = (5*player.card[3].stats['champ_id'] + 1)/300
    if lenght>4:
        all_stats[3][8] = (5*player.card[4].stats['champ_id'] + 1)/300
    for y in range(3):
        for x in range(7):
            if player.grid[y][x] is not None:
                all_stats[y+1][x] = (5*player.grid[y][x].stats['champ_id']+player.grid[y][x].level)/300
    for x in range(10):
        if player.bench[x] is not None:
            all_stats[0][x] = (5*player.bench[x].stats['champ_id']+player.bench[x].level)/300
    ens = []
    for en in players:
        if en != player:
            ens.append(en)
    for enm in range(1,8):
        for y in range(3):
            for x in range(7):
                if ens[enm-1].grid[y][x] is not None:
                    all_stats[y+(enm*4)+1][x] = (5*ens[enm-1].grid[y][x].stats['champ_id']+ens[enm-1].grid[y][x].level)/300
        all_stats[4*enm+1][7] = ens[enm-1].hp/100
        interest = ens[enm-1].gold//10
        if interest > 5: interest = 5.
        all_stats[4*enm+2][7] = interest/5
        all_stats[4*enm+3][7] = ens[enm-1].level/10
    return all_stats.reshape((1,1,32,10))



def convert_action(y_predict):
    y_predict[0] = 1.5*y_predict[0] + 0.5
    y_predict[2] = 1.5*y_predict[2] + 0.5
    y_predict[1] = 4.5*y_predict[1] + 4.5
    y_predict[3] = 4.5*y_predict[3] + 4.5
    return np.round(y_predict)


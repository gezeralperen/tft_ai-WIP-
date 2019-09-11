import numpy as np
import random

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_game_status(player, players):
    gold = player.gold / 50
    xp = player.xp / 100
    card1 = [0 for x in range(57)]
    try:
        card1[player.card[0].id] = 1
    except:
        card1[56] = 1
    card2 = [0 for x in range(57)]
    try:
        card2[player.card[1].id] = 1
    except:
        card2[56] = 1

    card3 = [0 for x in range(57)]
    try:
        card3[player.card[2].id] = 1
    except:
        card3[56] = 1

    card4 = [0 for x in range(57)]
    try:
        card4[player.card[3].id] = 1
    except:
        card4[56] = 1

    card5 = [0 for x in range(57)]
    try:
        card5[player.card[4].id] = 1
    except:
        card5[56] = 1

    player_card_map = card1 + card2 + card3 + card4 + card5
    player_champ_map = [0 for x in range(60 * 20)]
    for x in range(20):
        try:
            player_champ_map[(x * 60) + player.champions[x].id] = 1
            player_champ_map[(x * 60) + 57] = player.champions[x].level
            player_champ_map[(x * 60) + 58] = player.get_champ_location(player.champions[x])[1] / 2
            player_champ_map[(x * 60) + 59] = player.get_champ_location(player.champions[x])[0] / 6
        except:
            player_champ_map[(x * 60) + 56] = 1
    enemy_map = []
    for enemy in players:
        if enemy != player:
            gold = (enemy.gold // 10) / 5
            level = enemy.level / 9
            enemy_champ_map = [0 for x in range(60 * 20)]
            for x in range(20):
                try:
                    player_champ_map[(x * 60) + enemy.champions[x].id] = 1
                    player_champ_map[(x * 60) + 57] = enemy.champions[x].level
                    player_champ_map[(x * 60) + 58] = enemy.get_champ_location(enemy.champions[x])[1] / 2
                    player_champ_map[(x * 60) + 59] = player.get_champ_location(player.champions[x])[0] / 6
                except:
                    player_champ_map[(x * 60) + 56] = 1
            enemy_map += [gold] + [level] + enemy_champ_map

    player_map = [gold] + [xp] + player_card_map + player_champ_map + enemy_map
    return np.array(player_map)

def convert_action(y_predict):
    r = random.random()
    possibilities = softmax(y_predict[0][:28])
    c=0
    for a in possibilities:
        if r < a:
            selected_action = [c]
        else:
            r -= a
            c += 1
    if 6<selected_action[0]<27:
        xind, yind = 2* selected_action[0]+ 14, 2 * selected_action[0] + 15
        selected_action = np.around([selected_action[0], (y_predict[0][xind])*6, (y_predict[0][yind])*2])
    return selected_action


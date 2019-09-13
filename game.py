from database import champions_data, items_data, pool_counts, level_probabilities, xp_level
from interactions import set_stats
from decider import TD3
import random
from battle import initiate_fight
import timeit
from time import time
from io_array import get_game_status, convert_action
from copy import copy
import json
import numpy as np
import os

clear = lambda: os.system('cls')

PLAYER_COUNT = 8
INITIAL_HEALTH = 100
BENCH_LIMIT = 10
NULL_GRID = [[None, None, None, None, None, None, None], [None, None, None, None, None, None, None],
             [None, None, None, None, None, None, None]]

brain = TD3()


class Player:
    def __init__(self, pid):
        self.nick = 'Player' + str(pid)
        self.hp = INITIAL_HEALTH
        self.xp = 0
        self.gold = 0
        self.champions = []
        self.empty_items = []
        self.grid = [[None, None, None, None, None, None, None], [None, None, None, None, None, None, None],
                     [None, None, None, None, None, None, None]]
        self.bench = [None, None, None, None, None, None, None, None, None, None]
        self.alive = True
        self.card = None
        self.reward = 4
        self.history = []

        ##champ : Champion

    @property
    def level(self):
        for cxp in range(8):
            if self.xp < xp_level[cxp]:
                return cxp + 1
        return 9

    @property
    def on_grid(self):
        return 21 - len(self.get_empty())

    def get_card(self, card):
        self.card = card

    def get_champ(self, gotten_champ):
        # Can I afford this?
        if self.gold >= gotten_champ.stats['cost']:
            # Lets say that I have this
            self.champions.append(gotten_champ)
            self.gold -= gotten_champ.stats['cost']

            # Will it merge?
            if self.update_champs():
                return True

            # Will it fit in bench?
            elif self.bench[self.bench is None] is not None:
                self.bench[self.bench.index(None)] = gotten_champ
                return True

            # Is there space in grid?
            elif len(self.get_empty()) > 21 - self.level:
                cord = random.choice(self.get_empty())
                self.put_champ(gotten_champ, cord)
                return True
            else:
                self.champions.remove(gotten_champ)
                self.gold += gotten_champ.stats['cost']
                return False
        else:
            return False

    def update_champs(self):
        updated = False
        for the_champ in self.champions:
            repeat = 0
            indices = []
            for listed in self.champions:
                if the_champ.name == listed.name and the_champ.level == listed.level:
                    indices.append(self.champions.index(listed))
                    repeat += 1

            if repeat == 3:
                self.champions[indices[0]].level_up()
                will_delete = [self.champions[i] for i in indices[1:]]
                if will_delete[0] in self.bench: self.bench[self.bench.index(will_delete[0])] = None
                if will_delete[1] in self.bench: self.self.bench[self.bench.index(will_delete[1])] = None
                self.champions.remove(will_delete[0])
                self.champions.remove(will_delete[1])
                self.grid_check()
                updated = True
                break
        if updated: self.update_champs()
        return updated

    def get_champ_location(self, searched_champ):
        grid = self.find_in_grid(searched_champ)
        if grid[2]:
            return [['grid'], grid[0:2]]
        elif searched_champ in self.bench:
            return [['bench'], [-1, self.bench.index(searched_champ)]]
        else:
            return [['No match!']]

    def find_in_grid(self, any_champ):
        found = False
        count = 0
        cord1 = 0
        cord2 = 0

        for row in self.grid:
            if any_champ in row:
                cord2 = row.index(any_champ)
                cord1 = count
                found = True
            count += 1

        return [cord1, cord2, found]

    def grid_check(self):
        cy = 0
        for row in self.grid:
            cx = 0
            for value in row:
                if value != None and value not in self.champions:
                    self.grid[cy][cx] = None
                cx += 1
            cy += 1

    def get_empty(self):
        empty_list = []
        cord1 = 0
        cord2 = 0
        for row in self.grid:
            cord2 = 0
            for value in row:
                if value == None: empty_list.append([cord1, cord2])
                cord2 += 1
            cord1 += 1
        return empty_list

    def put_champ(self, champ, cord):
        if self.on_grid < self.level:
            loc = self.get_champ_location(champ)
            if cord[0] == -1 and loc[0] == 'grid':
                if self.bench[self.bench is None] is not None:
                    self.grid[loc[1][0]][loc[1][1]] = None
                    self.bench[self.bench.index(None)] = champ
                    return True
                else:
                    return False
            elif cord[1] < 7 and self.grid[cord[0]][cord[1]] == None:
                if loc == [['No match!']]:
                    if self.on_grid < self.level:
                        if champ in self.bench: self.bench[self.bench.index(champ)] = None
                        self.grid[cord[0]][cord[1]] = champ
                        return True
                    else:
                        return False
                if loc[0] == ['bench'] and 3 > cord[0] >= 0 and 7 > cord[0] >= 0:
                    self.grid[cord[0]][cord[1]] = champ
                    self.bench[self.bench.index(champ)] = None
                    return True
                elif loc[0] == ['grid'] and 3 > cord[0] >= 0 and 7 > cord[0] >= 0:
                    self.grid[cord[0]][cord[1]] = champ
                    self.grid[loc[1][0]][loc[1][1]] = None
                    return True
            elif cord[1] < 7:
                grid_champ = self.grid[cord[0]][cord[1]]
                if loc[0] == ['bench'] and 3 > cord[0] >= 0 and 7 > cord[0] >= 0:
                    self.grid[cord[0]][cord[1]] = champ
                    self.bench[self.bench.index(champ)] = grid_champ
                    return True
                elif loc[0] == ['grid'] and 3 > cord[0] >= 0 and 7 > cord[0] >= 0:
                    self.grid[cord[0]][cord[1]] = champ
                    self.grid[loc[1][0]][loc[1][1]] = grid_champ
                    return True
            return False
        else:
            return False

    def get_item(self, item_name):
        self.empty_items.append(item_name)

        # champ : Champion

    def sell_champ(self, champ, pool):
        if champ is not None:
            send_back_to_pool([champ], pool)
            self.empty_items.extend(champ.items)
            self.gold += champ.stats['cost']
            self.champions.remove(champ)
            if champ in self.bench:
                self.bench[self.bench.index(champ)] = None
            self.grid_check()
            return 0, True
        else:
            return -4, True

        # dmg: integer

    def attacked(self, dmg):
        self.hp -= dmg
        if self.hp < 1:
            self.alive = False

    def fight_ready(self):
        self.grid_check()
        if self.on_grid < self.level and self.bench[self.bench is not None] is not None:
            self.put_champ(random.choice(self.bench), random.choice(self.get_empty()))
            self.fight_ready()
        for every_champ in self.champions:
            every_champ.fight_ready()


class Champion:
    def __init__(self, champ_name):
        self.name = champ_name
        self.level = 1
        self.items = []
        self.stats = set_stats(champ_name, self.items, self.level)
        self.champ_id = random.randint(0, 1000000000)
        self.owner = None
        self.fight_hp = self.stats["health"][self.level - 1]
        self.fight_mana = self.stats["starting_mana"]
        self.target = None
        self.last_interaction = 0
        self.next_interaction = 0
        self.isAlive = True

    def __str__(self):
        return f'{self.level}-{self.name}'

    def fight_ready(self):
        self.target = None
        self.fight_hp = self.stats['health'][self.level - 1]
        self.fight_mana = self.stats['starting_mana']
        self.last_interaction = 0
        self.next_interaction = 0
        self.isAlive = True

    def set_item(self, item_name):
        for item in self.items:
            if (item and item_name) in items_data['basic']:
                # item_added = item_merge(item, item_name)
                item_added = item_name
                self.items.remove(item)
                self.items.append(item_added)
                self.stats = set_stats(self.name, self.items, self.level)
                return True
        if len(self.items) == 3:
            return False

        self.items.append(item_name)

    def level_up(self):
        if self.level < 3:
            self.level += 1
            return True
        else:
            return False

    __repr__ = __str__


# INITIALIZE THE GAME  ########################


###############################################


def champ_card(level, pool):
    card = []
    chances = level_probabilities[str(level)].copy()
    index = 0
    for x in range(4):
        chances[x + 1] += chances[x]
    for x in range(5):
        choice = None
        while choice is None:
            try:
                prob = random.random()
                if prob < chances[0] and len(pool[0]) > 0:
                    index = 0
                    choice = random.choice(pool[0])
                    card.append(choice)
                elif prob < chances[1] and len(pool[1]) > 0:
                    index = 1
                    choice = random.choice(pool[1])
                    card.append(choice)
                elif prob < chances[2] and len(pool[2]) > 0:
                    index = 2
                    choice = random.choice(pool[2])
                    card.append(choice)
                elif prob < chances[3] and len(pool[3]) > 0:
                    index = 3
                    choice = random.choice(pool[3])
                    card.append(choice)
                else:
                    index = 4
                    choice = random.choice(pool[4])
                    card.append(choice)
            except IndexError:
                continue
        pool[index].remove(choice)

    return card


def send_back_to_pool(card, pool):
    for champ in card:
        pool[champ.stats['cost'] - 1].append(champ)
        card.remove(champ)


def new_cards(player, pool):
    if player.card is not None:
        send_back_to_pool(player.card, pool)
    player.card = champ_card(player.level, pool)


def select_from_card(player, index):
    if index < len(player.card):
        check = player.get_champ(player.card[index])
        if check:
            # print(f'You picked {player.card[index]}')
            player.card.remove(player.card[index])
            return True
        else:
            return False
    return False


def take_action(response, player, pool):
    def select_card(id):
        if len(player.card) > id:
            selected = select_from_card(player, id)
            if selected:
                # print(f'{player.nick} added {player.champions[-1]} to its hand.')
                return 0, True
            else:
                return -4, True
        else:
            return -4, True

    y, x, a, b = response.round().astype(dtype='int')
    # print(f'y: {y}\tx: {x}\ta: {a}\tb: {b}')
    if -1 < y < 3:
        if x < 7:
            champ = player.grid[y][x]
            if a == 2 and b == 9:
                return player.sell_champ(champ, pool)
            if player.put_champ(champ, [a, b]):
                return player.sell_champ(champ, pool)
            else:
                return -4, True
        if x > 7:
            if y == 0 and x == 8:
                return select_card(0)
            if y == 0 and x == 9:
                return select_card(1)
            if y == 1 and x == 8:
                return select_card(2)
            if y == 1 and x == 9:
                return select_card(3)
            if y == 2 and x == 8:
                return select_card(4)
        if x == 7:
            if y == 0:
                # print(f'({player.level}) {player.nick} finished it\'s round.')
                player.xp += 2
                return 0, False
            if y == 1:
                if player.gold >= 2:
                    new_cards(player, pool)
                    player.gold -= 2
                    # print(f'({player.level}) {player.nick} got new cards.')
                    return 0, True
                else:
                    return -4, True
            if y == 2:
                if player.gold > 3:
                    # print(f'({player.level}) {player.nick} bought Xp.')
                    player.xp += 4
                    player.gold -= 4
                    return 0, True
                else:
                    return -4, True
            if y == -1:
                champ = player.bench[x]
                player.put_champ(champ, [a, b])
                return 0, True
    if y == -1:
        champ = player.bench[y]
        if a == 2 and b == 9:
            return player.sell_champ(champ, pool)
        check = player.put_champ(champ, [a, b])
        if check: return 0, True
        return -4, True
    return -4, True


def play_round(player, pool, players):
    if player.alive:
        # print(f'It\'s the turn of {player.nick}')
        new_cards(player, pool)
        # print(f'Gold: {player.gold} \t Xp: {player.xp} \t Champions: \n{player.champions}\nCards:')
        play = True
        played = 0
        while play and played < 50:
            played += 1
            # print(f'{player.nick} -- Played {played} times.')
            stats = get_game_status(player, players)
            action = brain.select_action(stats)
            # action = np.random.uniform(-1,1,4)
            response = convert_action(action.copy())
            reward, play = take_action(response, player, pool)
            next_stats = get_game_status(player, players)
            response = np.array(response)
            player.history.append({'State': stats.tolist(), 'Next State': next_stats.tolist(),
                                   'Action': action.tolist(), 'Reward': reward, 'Done': 0, 'Q': 0})
        return reward


def print_player_stats(player):
    print(f'''

                __{player.nick}__
    Gold : {player.gold}    Level: {player.level}   Xp: {player.xp} Hp: {player.hp} Reward: {player.reward}''')
    print('==================================================')
    for row in player.grid:
        print(row)
    print('--------------------------------------------------')
    print(player.bench)
    print('--------------------------------------------------')
    print(f'Champion count : {len(player.champions)}\t Bench Count : {len(player.bench)}\t Grid Count :'
          f' {21 - len(player.get_empty())}')
    print('==================================================')


if __name__ == '__main__':
    brain.load('last_save', 'models')
    while True:
        for x in range(5):
            start = timeit.default_timer()
            pool = [[], [], [], [], []]
            for champ in champions_data:
                for cst in range(1, 6):
                    if champions_data[champ]['cost'] == cst:
                        for x in range(pool_counts[str(cst)]):
                            pool[cst - 1].append(Champion(str(champ)))
                        break
            players = [Player(x + 1) for x in range(PLAYER_COUNT)]

            for player in players:
                player.gold = 5
                player.xp = 19
                new_cards(player, pool)
                select_from_card(player, 0)
                send_back_to_pool(player.card, pool)
                player.gold = 0
                player.xp = 2

            game_over = False
            x = 0
            while not game_over:
                x += 1
                print(f'{x}. Turn')
                alive_players = [a for a in players if a.alive]
                random.shuffle(alive_players)
                for player in alive_players:
                    interest = player.gold // 10
                    if interest > 5: interest = 5
                    player.gold += 5 + interest
                    player.xp += 2
                    round_reward = play_round(player, pool, players)
                    player.fight_ready()

                if len(alive_players) % 2 == 1:
                    bot = copy(random.choice(alive_players))
                    bot.nick += '_bot'
                    initiate_fight(alive_players[len(alive_players) // 2 + 1], bot)
                    if not alive_players[len(alive_players) // 2 + 1].alive:
                        alive_players[len(alive_players) // 2 + 1].reward = 5 - len(alive_players)

                for pairs in range(int(len(alive_players) // 2)):
                    initiate_fight(alive_players[pairs], alive_players[-pairs - 1])
                    if not alive_players[pairs].alive:
                        alive_players[pairs].reward = 5 - len(alive_players)
                    if not alive_players[-1 - pairs].alive:
                        alive_players[-1 - pairs].reward = 5 - len(alive_players)

                if len([a for a in players if a.alive]) < 2:
                    game_over = True
            stop = timeit.default_timer()
            print('Time: ', stop - start)

            for player in players:
                player.history[-1]['Reward'] = player.reward
                player.history[-1]['Done'] = 1
                player.history[-1]['Q'] = player.reward
                for x in range(len(player.history) - 1):
                    if player.history[-1 - x]['Reward'] != -4:
                        player.history[-2 - x]['Q'] = player.history[-2 - x]['Reward'] + 0.99 * player.history[-1 - x]['Q']
                    else:
                        i = 0
                        while player.history[-1 - x + i]['Reward'] == -4:
                            i += 1
                        player.history[-2 - x]['Q'] = player.history[-2 - x]['Reward'] + 0.99 * player.history[-1 - x + i][
                            'Q']
                file = open(f'replay_experience/{time()}.json', 'w')
                json.dump(player.history, file)
                file.close()
                print_player_stats(player)
        for x in range(5):
            brain.train(3)
            brain.save('last_save', 'models')
        clear()

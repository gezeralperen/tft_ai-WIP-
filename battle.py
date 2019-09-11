import numpy as np
import random


def rotate180(enms):
    new_array = [[None for ix in range(7)] for vy in range(3)]
    for y in range(len(enms)):
        for x in range(len(enms[y])):
            new_array[-y - 1][-1 - x] = enms[y][x]
    return new_array


def get_adj(id, power=1):
    row = int(id / 7)
    indices = []
    unique = []
    if row == 0:
        if id % 7 == 0:
            indices.extend([id + 7, id + 1])
        elif id % 7 == 6:
            indices.extend([id + 7, id + 6, id - 1])
        else:
            indices.extend([id + 6, id + 7, id - 1, id + 1])
    elif row == 5:
        if id % 7 == 0:
            indices.extend([id - 7, id - 6, id + 1])
        elif id % 7 == 6:
            indices.extend([id - 7, id - 1])
        else:
            indices.extend([id - 7, id - 6, id - 1, id + 1])
    else:
        if row % 2 == 1:
            if id % 7 == 0:
                indices.extend([id - 7, id - 6, id + 1, id + 7, id + 8])
            elif id % 7 == 6:
                indices.extend([id - 7, id - 1, id + 7])
            else:
                indices.extend([id - 7, id - 6, id - 1, id + 1, id + 7, id + 8])
        elif row % 2 == 0:
            if id % 7 == 0:
                indices.extend([id - 7, id + 1, id + 7])
            elif id % 7 == 6:
                indices.extend([id - 8, id - 7, id - 1, id + 6, id + 7])
            else:
                indices.extend([id - 8, id - 7, id - 1, id + 1, id + 6, id + 7])
    powers = []

    if power > 1:
        for ind in indices:
            powers.extend(get_adj(ind, power - 1))
        indices.extend(powers)

    for x in indices:
        if x not in unique and x != id:
            unique.append(x)

    return unique


def cord_to_id(cord):
    return cord[0] * 7 + cord[1]


def id_to_cord(id):
    return [int(id / 7), id % 7]


class Grid:  # Environment
    def __init__(self, full_grid, you):
        self.cord = cord_to_id(you)
        self.full_grid = full_grid

    def set(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def move(self, action):
        # check if legal move first
        if action in self.actions[self.cord]:
            self.cord = action
        # return a reward (if any)
        return self.rewards[self.cord]

    def get_states(self):
        states = []
        x=0
        for row in self.full_grid:
            for each in row:
                if each is None:
                    states.append(x)
                x += 1
        return states


def get_empty(grid):
    empty_list = []
    cord1 = 0
    for row in grid:
        cord2 = 0
        for value in row:
            if value is None: empty_list.append([cord1, cord2])
            cord2 += 1
        cord1 += 1
    return empty_list


def standard_grid(full_grid, you, enemy_tag, power=1):
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state


    g = Grid(full_grid, you)
    target = []

    y = 0
    for row in full_grid:
        x = 0
        for each in row:
            if each == enemy_tag:
                target.extend(get_adj(cord_to_id([y, x]), power))
            x += 1
        y += 1

    rewards = [-0.5 for x in range(42)]
    for index in target:
        rewards[index] = 1
    actions = [None] * 42
    for cid in range(42):
        deleted = []
        adjs = get_adj(cid, 1)
        for vst in adjs:
            if full_grid[id_to_cord(vst)[0]][id_to_cord(vst)[1]] is not None:
                deleted.append(vst)
        for every in deleted:
            adjs.remove(every)
        if len(adjs) == 0:
            adjs = [None]
        actions[cid] = adjs

    g.set(rewards, actions)
    return g


def print_grid(any_list):
    print('|------------------------------------|')
    for x in range(6):
        for y in range(7):
            print(f'| {any_list[x * 7 + y]} ', end='')
        print('|\n|------------------------------------|')


SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def optimal_step(full_grid, you, enemy_tag, c_range):
    grid = standard_grid(full_grid, you, enemy_tag, power=c_range)
    policy = [random.choice(x) for x in grid.actions]

    V = [0 for x in range(42)]
    states = grid.get_states()

    # repeat until convergence
    # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            # V(s) only has value if it's not a terminal state
            if s in policy:
                new_v = float('-inf')
                for a in grid.actions[s]:
                    if a is None: continue
                    grid.cord = s
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.cord]
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break

    # find a policy that leads to optimal value function
    s = cord_to_id(you)
    best_a = None
    best_value = float('-inf')
    # loop through all possible actions to find the best current action
    for a in grid.actions[s]:
        if a is None: continue
        grid.cord = s
        r = grid.move(a)
        v = r + GAMMA * V[grid.cord]
        if v > best_value:
            best_value = v
            best_a = a

    return best_a


def set_owner(player_nick, grid):
    ret_array = []
    for row in grid:
        new_row = []
        for each in row:
            if each is not None:
                each.owner = player_nick
            new_row.append(each)
        ret_array.append(new_row)
    return ret_array


def search_for_enemy(hex_id, range, grid, enemy_tag):
    enemies_in_range = []
    adj_tiles = [id_to_cord(x) for x in get_adj(hex_id, power=range)]

    for tile in adj_tiles:
        if grid[tile[0]][tile[1]] is not None and  grid[tile[0]][tile[1]].owner == enemy_tag:
            enemies_in_range.append(tile)
    if len(enemies_in_range) == 0:
        go_to = optimal_step(grid, id_to_cord(hex_id), enemy_tag, c_range=range)
        if go_to is None:
            go_to = hex_id
        return id_to_cord(go_to), False
    else:
        return random.choice(enemies_in_range), True

def count_champs(grid, player_nick):
    count = 0
    for row in grid:
        for hexes in row:
            if hexes is not None and hexes.owner == player_nick and hexes.fight_hp > 0: count += 1
    return count


def champ_attack(c_champ, will_be_attacked):
    will_be_attacked.fight_hp -= c_champ.stats['damage'][c_champ.level-1]
    if will_be_attacked.fight_hp <= 0:
        will_be_attacked.isAlive = False

def initiate_ult(cchamp, hex_id):
    pass

def initiate_fight(player1, player2):
    print(f'{player1.nick} vs {player2.nick}')
    team1_grid = []
    team1_grid.extend(player1.grid)
    team1_grid = set_owner(player1.nick, team1_grid)

    team2_grid = []
    team2_grid.extend(player2.grid)
    team2_grid = set_owner(player2.nick, team2_grid)

    team2_grid = rotate180(team2_grid)

    team2_grid.extend(team1_grid)

    final_grid = team2_grid

    t=0
    battle_end = False
    while not battle_end and t < 6000:
        hex_id = 0
        played_this_t = []
        for row in final_grid:
            for hex in row:
                if hex is not None and hex not in played_this_t:
                    if hex.fight_hp <= 0:
                        y = 0
                        for y in range(len(final_grid)):
                            chk = False
                            x = 0
                            for x in range(y):
                                if final_grid[y][x] == hex:
                                    final_grid[y][x] = None
                                    hex.isAlive = False
                                    played_this_t.append(hex)
                                    chk = True
                                    break
                                x += 1
                            if chk: break
                            y += 1
                        continue

                    if t >= hex.next_interaction:
                        if hex.target not in get_adj(hex_id, hex.stats['range']) and (hex.target is not None and not hex.target.isAlive):
                            hex.target = None
                        if hex.target is None:
                            if player2.nick == hex.owner:
                                enemy_player = player1.nick
                            else:
                                enemy_player = player2.nick
                            final_grid[int(hex_id/7)][hex_id%7] = None
                            response, found = search_for_enemy(hex_id, hex.stats['range'], final_grid, enemy_player)

                            if found:
                                hex.target = final_grid[response[0]][response[1]]
                            else:
                                final_grid[response[0]][response[1]] = hex
                                final_grid[int(hex_id/7)][hex_id%7] = None
                                played_this_t.append(hex)
                                hex.last_interaction = t
                                hex.next_interaction = t+50

                        if hex.target is not None:
                            final_grid[int(hex_id/7)][hex_id%7] = hex

                            champ_attack(hex,hex.target)
                            hex.fight_mana += 10
                            hex.last_interaction = t
                            hex.next_interaction = t+hex.stats['speed']
                            if hex.fight_mana >= hex.stats['mana']:
                                initiate_ult(hex, hex_id)
                            if hex.target.isAlive:
                                hex.target.fight_mana += 10
                                if hex.target.fight_mana >= hex.target.stats['mana']:
                                    initiate_ult(hex.target, hex_id)
                            else:
                                y = 0
                                for y in range(len(final_grid)):
                                    x = 0
                                    chk = False
                                    for x in range(y):
                                        if final_grid[y][x] == hex.target:
                                            final_grid[y][x] = None
                                            chk = True
                                            break
                                        x += 1
                                    y +=1
                                    if chk: break

                hex_id += 1
        if count_champs(final_grid, player1.nick) == 0:
            winner = player2.nick
            player1.attacked(5)
            battle_end = True
        if count_champs(final_grid, player2.nick) == 0:
            winner = player1.nick
            player2.attacked(5)
            battle_end = True
        t += 1
    if not battle_end:
        winner = 'Noone'
        player1.attacked(5)
        player2.attacked(5)
    print(f'Winner is {winner}!')
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import json

browser = webdriver.Chrome()

####

URL = "https://tftactics.gg/champions"
browser.get(URL)
while True:
    try:
        if len(browser.find_elements_by_class_name("characters-list")) > 0:
            break
        else:
            time.sleep(1)
    except:
        time.sleep(1)

r = browser.execute_script("return document.body.innerHTML")
soup = BeautifulSoup(r, 'lxml')
champlist = soup.find('div', class_= 'characters-list')
cost1 = champlist.find_all(text=True)

champ_db = {}

cid = 0
for champ in cost1:
    url = f'https://tftactics.gg/champions/{champ.lower().replace(" ", "_")}'
    browser.get(url)
    while True:
        try:
            if len(browser.find_elements_by_class_name("stats-list")) > 0:
                break
            else:
                time.sleep(1)
        except:
            time.sleep(1)
    r = browser.execute_script("return document.body.innerHTML")
    soup = BeautifulSoup(r, 'lxml')
    stats = soup.find('ul', class_='stats-list').find_all(text=True)
    if 'Starting Mana: ' in stats:
        champ_stat = {
            'cost': int(stats[1]),
            'champ_id': cid,
            'health': [int(i) for i in stats[3].split(' / ')],
            'mana': int(stats[5]),
            'starting_mana': int(stats[7]),
            'armor': int(stats[9]),
            'mr': int(stats[11]),
            'damage': [int(i) for i in stats[15].split(' / ')],
            'speed': int(100*(1/float(stats[17]))),
            'crit': float(stats[19][:-1])/100,
            'range': int(stats[21])
        }
    else:
        champ_stat = {
            'cost': int(stats[1]),
            'champ_id': cid,
            'health': [int(i) for i in stats[3].split(' / ')],
            'mana': int(stats[5]),
            'starting_mana': 0,
            'armor': int(stats[7]),
            'mr': int(stats[9]),
            'damage': [int(i) for i in stats[13].split(' / ')],
            'speed': int(100*(1/float(stats[15]))),
            'crit': float(stats[17][:-1])/100,
            'range': int(stats[19])
        }
    origins = soup.find_all('div', class_='character-ability')
    origins = [i.find('h2').text for i in origins[1:]]
    champ_stat.update({'origins': origins})
    champ_stat = json.dumps(champ_stat)
    champ_stat = '{' f'"{champ}": ' + champ_stat + '}'
    champ_stat = json.loads(champ_stat)
    champ_db.update(champ_stat)
    cid += 1



json.dump(champ_db, open('champions_data.json', 'w'))




browser.quit()



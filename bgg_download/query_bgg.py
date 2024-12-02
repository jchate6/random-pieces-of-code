import requests
import xml.etree.ElementTree as ET
import csv
import argparse


class XmlDictConfig(dict):
    """
    Note: need to add a root into if no exising
    Example usage:
    tree = ElementTree.parse('your_file.xml')
    root = tree.getroot()
    xmldict = XmlDictConfig(root)
    Or, if you want to use an XML string:
    root = ElementTree.XML(xml_string)
    xmldict = XmlDictConfig(root)
    And then use xmldict for what it is... a dict.
    """
    def __init__(self, parent_element):
        if parent_element.items():
            self.updateShim( dict(parent_element.items()) )
        for element in parent_element:
            if len(element):
                aDict = XmlDictConfig(element)
            #   if element.items():
            #   aDict.updateShim(dict(element.items()))
                self.updateShim({element.tag: aDict})
            elif element.items():    # items() is specialy for attribtes
                elementattrib= element.items()
                if element.text:
                    elementattrib.append((element.tag,element.text ))     # add tag:text if there exist
                self.updateShim({element.tag: dict(elementattrib)})
            else:
                self.updateShim({element.tag: element.text})

    def updateShim (self, aDict ):
        for key in aDict.keys():   # keys() includes tag and attributes
            if key in self:
                value = self.pop(key)
                if type(value) is not list:
                    listOfDicts = []
                    listOfDicts.append(value)
                    listOfDicts.append(aDict[key])
                    self.update({key: listOfDicts})
                else:
                    value.append(aDict[key])
                    self.update({key: value})
            else:
                self.update({key: aDict[key]})  # it was self.update(aDict)


def get_correct_game_id(game_name, limit=10, try_hard=False):

    if try_hard:
        th_name = game_name.split('-')[0].split('(')[0]
        params = {'search': th_name}
    else:
        params = {'search': game_name}

    response = requests.get(url=search_url, params=params)

    root = ET.XML(response.text)

    root_dict = XmlDictConfig(root)

    print('==================================================')
    print(f'Games Matching "{game_name}":')
    matching_boardgames = root_dict.get('boardgame')
    if matching_boardgames is None:
        print('No games found.')
        return None
    if not isinstance(matching_boardgames, list):
        matching_boardgames = [matching_boardgames]
    for i, game in enumerate(matching_boardgames):
        try:
            print(f'{i}: {game["name"]["name"]} ({game.get("yearpublished")})')
        except TypeError:
            print(f'{i}: {game["name"]} ({game.get("yearpublished")})')
        if i >= limit:
            break
    print('n: None of the above')

    prompt = f'Which version of {game_name} is correct? (0)'
    while True:
        response = input(prompt)
        if response == '':
            response = 0
        if response == 'n':
            return None
        try:
            games_id = matching_boardgames[int(response)]['objectid']
            break
        except (IndexError, ValueError):
            print('Invalid response. Please try again.')
    return games_id


def get_game_data(games):
    game_data = []
    for game_id in games:
        game_response = requests.get(url=f'{game_url}/{game_id}', params={'stats': 1})
        game_root = ET.XML(game_response.text)
        game_dict = XmlDictConfig(game_root)

        data_dict = {
            'min_players': game_dict['boardgame']['minplayers'],
            'max_players': game_dict['boardgame']['maxplayers'],
            'min_playtime': game_dict['boardgame']['minplaytime'],
            'max_playtime': game_dict['boardgame']['maxplaytime'],
            'min_age': game_dict['boardgame']['age'],
            'complexity': game_dict['boardgame']['statistics']['ratings']['averageweight'],
            'rating': game_dict['boardgame']['statistics']['ratings']['average'],
        }
        for poll in game_dict['boardgame']['poll-summary']['result']:
            if poll['name'] == 'bestwith':
                best_players = poll['value'].replace('Best with ', '').replace(' players', '')
                break
            else:
                best_players = None
        else:
            best_players = None
        data_dict['best_players'] = best_players
        tags = []
        types_list = game_dict['boardgame']['statistics']['ratings']['ranks']['rank']
        if not isinstance(types_list, list):
            types_list = [types_list]
        for bg_type in types_list:
            if bg_type['name'] != 'boardgame':
                tags.append(bg_type['name'].replace('games', ''))
        data_dict['tags'] = ', '.join(tags)
        name_list = game_dict['boardgame']['name']
        if not isinstance(name_list, list):
            name_list = [name_list]
        for name in name_list:
            if name.get('primary') == 'true':
                primary_name = name.get('name')
        data_dict['name'] = primary_name
        game_data.append(data_dict)
    return game_data


def write_csv(game_data):
    with open('bgg_outputs.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'min_players', 'max_players', 'best_players', 'complexity', 'min_playtime', 'max_playtime',
                      'min_age', 'rating', 'tags']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(game_data)


def open_csv(input_file):
    games_list = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for lines in reader:
            games_list.append(lines[0])
    return games_list


def save_ids(filename, game_id):
    with open(filename, 'a') as f:
        f.write(game_id + '\n')


def clear_file(filename):
    with open(filename, 'w') as f:
        f.write('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", help="Clear File Cache", default=False, action="store_true")
    parser.add_argument("--th", help="try hard: remove details during search", default=False, action="store_true")
    args = parser.parse_args()
    try_hard = args.th

    input_file = 'input_list.csv'
    # input_file = 'new_input.csv'
    done_id_cache_file = 'done_id_cache.txt'
    done_name_cache_file = 'done_name_cache.txt'
    bad_cache_file = 'bad_cache.txt'
    base_url = 'https://www.boardgamegeek.com/xmlapi'
    search_url = base_url + '/search'
    game_url = base_url + '/boardgame'

    if args.cc:
        clear_file(done_id_cache_file)
        clear_file(done_name_cache_file)
        clear_file(bad_cache_file)

    games_list = open_csv(input_file)
    # games_list = ['Barenpark', 'A Game of Thrones the Board Game - A Dance with Dragons expansion', '7 Wonders Duel', 'Carcason']
    done_list = open_csv(done_name_cache_file)
    bad_list = open_csv(bad_cache_file)
    for game in done_list:
        try:
            games_list.remove(game)
        except ValueError:
            pass
    for game in bad_list:
        try:
            games_list.remove(game)
        except ValueError:
            pass

    game_ids = open_csv(done_id_cache_file)
    no_game_found = []
    for game_name in games_list:
        game_id = get_correct_game_id(game_name, try_hard=try_hard)
        if game_id is None:
            game_id = get_correct_game_id(game_name, 1000, try_hard=try_hard)
        if game_id is not None:
            game_ids.append(game_id)
            save_ids(done_id_cache_file, game_id)
            save_ids(done_name_cache_file, game_name)
        else:
            no_game_found.append(game_name)
            save_ids(bad_cache_file, game_name)

    game_data = get_game_data(game_ids)
    write_csv(game_data)
    print(no_game_found)

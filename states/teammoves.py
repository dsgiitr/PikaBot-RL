from poke_env import Player, ShowdownServerConfiguration, AccountConfiguration
import asyncio
import json
import copy
from names import name_operate


import generate_states
state=generate_states.DamageCalculator()

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # if battle.available_moves:
        #     best_move = max(battle.available_moves, key=lambda move: move.base_power)

        #     if battle.can_tera:
        #         return self.create_order(best_move, terastallize=True)

        #     return self.create_order(best_move)
        # else:
            return self.choose_random_move(battle)


class RandomPlayerMine(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_events = []
        self.last_turn = 0
        self.pokemon_data = [[],{}]
        self.logs=[]
        with open("monmap1.json", 'r') as fh:
            self.mons = json.load(fh)
        self.items={}
        self.moves={}
    
    def updateData(self, battle):
        # state.getMons(battle)
        # state.getOppMons(battle)
        move_used = ""
        isattack = False
        attacker = ""
        
        self.pokemon_data[0] = [battle.active_pokemon,]
        self.pokemon_data[0].extend(battle.available_switches)
        OppName=name_operate.getName(self.mons, battle.opponent_active_pokemon)
        if OppName not in self.pokemon_data[1]:
            self.pokemon_data[1][OppName] = [{}, 100, name_operate.species(battle.opponent_active_pokemon),{}]
        move_used=""
        if battle.turn > 1:
            for event in battle.observations[battle.turn-1].events:
                if(event[1]=="-boost") and (event[2].split(':')[0] == "p2a"):
                    mon=name_operate.convertName(event[2].split())
                    mons=name_operate.searchMon(mon,self.pokemon_data)
                    if event[3] not in self.pokemon_data[1][mons][3]:
                        self.pokemon_data[1][mons][3][event[3]]=int(event[4])
                    else:
                        self.pokemon_data[1][mons][3][event[3]]+=int(event[4])
                elif (event[1] == "move") and (event[2].split(':')[0] == "p2a"):  # Check if it's a move by the opponent
                    move_used = event[3]  # Extract the move used
                    mon=name_operate.convertName(event[2].split())
                    attacker=name_operate.searchMon(mon,self.pokemon_data)
                elif event[1] == "-damage" and event[2].split(':')[0] == "p1a" and len(event) == 4:
                    isattack = True
            for event in battle.observations[battle.turn-1].events:
                if event[1] == "-damage" or event[1] == "-heal":
                    if event[2].split(':')[0] == 'p2a':
                        mon=name_operate.convertName(event[2].split())
                        mons=name_operate.searchMon(mon,self.pokemon_data)
                        if(mons!="None"):
                            if len(event[3].split('/')) == 1:
                                self.pokemon_data[1][mons][1] = int(event[3].split()[0])
                            else:
                                self.pokemon_data[1][mons][1] = int(event[3].split('/')[0])
                            break
        
        if move_used != "" and attacker in self.pokemon_data[1]:
                self.pokemon_data[1][attacker][0][name_operate.operation(move_used)] = isattack
        # newlog = copy.deepcopy(self.pokemon_data[1])
        # self.logs.append([battle.turn-1, newlog])
    
    def choose_move(self, battle):
        # print(battle.turn)
        self.last_turn += 1
        if self.last_turn == battle.turn:
            self.updateData(battle)
            State_space=state.states(self.pokemon_data,battle,self.mons)
            print(State_space)
        else:
            self.last_turn -=1
        return self.choose_random_move(battle)
    
    def _battle_finished_callback(self, battle):
         self.pokemon_data=[[],{}]



async def run_battle(player, opponent):
    await player.battle_against(opponent, n_battles=1)

async def main():
    # account_config = AccountConfiguration("tushar230423042304", "iamtushar")
    # random_player = RandomPlayerMine(server_configuration=ShowdownServerConfiguration, account_configuration=account_config)
    random_player = RandomPlayerMine(battle_format='gen9randombattle')
    second_player = MaxDamagePlayer(battle_format='gen9randombattle')
    for i in range(1):
        await run_battle(random_player, second_player)
    # await random_player.send_challenges("tushar230423042304", n_challenges=1)
    # await random_player.accept_challenges(None, 1)
    return random_player, second_player

async def main2():
    random_player, second_player = await main()
    print(f"Player {random_player.username} won {random_player.n_won_battles} out of {random_player.n_finished_battles} played")
    print(f"Player {second_player.username} won {second_player.n_won_battles} out of {second_player.n_finished_battles} played")
    logs = random_player.logs
    # for mon in logs[-1][1][0]:
    #     print(mon)
    # for mon in logs[-1][1][1]:
    #     print(mon)
    # newmons=random_player.mons
    # with open('battle_data.json', 'w') as f:
    #     json.dump(logs, f, indent=4)
    # with open('movemapu1.json','w') as f:
    #     json.dump(newmons,f)



asyncio.run(main2())
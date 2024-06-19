from poke_env import Player, ShowdownServerConfiguration, AccountConfiguration
# import asyncio
# import json
# import copy
# import generate_states
import requests
import json
from names import name_operate

class DamageCalculator:
    def __init__(self):
        self.OpponentMons=[]
        self.OurMons=[]
        self.used=False
        self.P1_mon={}
        self.OppLastMon=""
        self.P2_mon={}
        self.itemdict={}
        with open('movemap1.json','r') as f:
            self.movedict=json.load(f)
        with open('itemmap1.json','r') as f:
            self.itemname=json.load(f)
        self.itemname['']=''
        with open('poke_battle.json','r') as f:
            temp=json.load(f)
            for mon in temp:
                if 'items' in temp[mon]:
                    self.itemdict[mon]=temp[mon]['items'][0]
                else:
                    self.itemdict[mon]=""
            self.leveldict={mon:temp[mon]['level'] for mon in temp}
            self.data=temp
        with open("HP_Speed.json",'r') as f:
            self.base_stat=json.load(f)
            
    #1st part

    def find_role(self,pokemon_name, moves, item=None):
        if pokemon_name in self.data:
            pokemon = self.data[pokemon_name]
            for role_name, role_data in pokemon["roles"].items():
                if len(moves)==0:
                    if item is None or item in role_data["items"]:
                        return role_name
                for move in moves:
                    if move in role_data["moves"]:
                        if item is None or item in role_data["items"]:
                            return role_name
        return "Role not found"


    # 2nd part

    def get_moves_for_role(self,pokemon_name, role_name):
        moves = []
        if pokemon_name in self.data:
            pokemon = self.data[pokemon_name]
            if "roles" in pokemon and role_name in pokemon["roles"]:
                moves.extend(pokemon["roles"][role_name]["moves"])
        return moves

    #3rd part

    def move_state(self,list1, list2):
        list3 = list(list2)  # Start with elements from list2
        remaining_space = 4 - len(list3)  # Calculate remaining space in list3

        # Add elements from list1 that are not already in list2
        for element in list1:
            if element not in list3 and remaining_space > 0:
                list3.append(element)
                remaining_space -= 1

        return list3[:4]  # Ensure list3 has a maximum size of 4
    
    @staticmethod
    def calculate(params):
        url = "http://localhost:5000/calculate"
        response=requests.get(url,params=params) # Sends the request to the server to calculate
        if response.status_code == 200:
            result = response.json() # result
            if(type(result)==int):
                return result #if 0, then return 0
            a=0
            for x in result:
                a+=x
            a/=16
            return (int)(a) #return average
        else:
            print(f"Error: {response.status_code}")
            return 0
    
    def DamageCalc(self,monA,defender,mon,move,allmons):
        param={}
        if(str(move.category)!="STATUS (move category) object"):
            attacker=name_operate.getName(allmons,monA)
            param["attacker"]=attacker
            param["defender"]=defender
            param["move"]=self.movedict[move.id]
            param["level1"]=self.leveldict[attacker]
            param["level2"]=self.leveldict[defender]
            d1={}
            for i in monA.boosts:
                if(i!="accuracy" and i!="evasion" and monA.boosts[i]!=0):
                    d1[i]=monA.boosts[i]
            param["boosts1"]=json.dumps(d1)
            d2={}
            for i in mon[3]:
                if(i!="accuracy" and i!="evasion" and mon[3][i]!=0):
                    d2[i]=mon[3][i]
            param["boosts2"]=json.dumps(d2)
            if monA.item!=None:
                param['item1']=self.itemname[monA.item]
            param['item2']=self.itemdict[defender]
            
            return DamageCalculator.calculate(param)
        return -1

    def Opponent_DamageCalc(self,name,monA,move,allmons,boosts):
        param={}
        attacker=name
        defender=name_operate.getName(allmons,monA)
        param["attacker"]=attacker
        param["defender"]=defender
        param["move"]=move
        param["level1"]=self.leveldict[attacker]
        param["level2"]=self.leveldict[defender]
        d2={}
        for i in boosts:
            if(i!="accuracy" and i!="evasion" and boosts[i]!=0):
                d2[i]=boosts[i]
        param["boosts1"]=json.dumps(d2)
        d1={}
        for i in monA.boosts:
            if(i!="accuracy" and i!="evasion" and monA.boosts[i]!=0):
                d1[i]=monA.boosts[i]
        param["boosts2"]=json.dumps(d1)
        if monA.item!=None:
            param['item2']=self.itemname[monA.item]
        param['item1']=self.itemdict[defender]
        
        return DamageCalculator.calculate(param)

    def damage_to_state(dmg,curhp):
        curhp=int(curhp)
        if(dmg<=0):
            return dmg
        return min(5,int((curhp+dmg-1)/dmg))
    def checkSpeed(self,allmons,monA,mon):
        boosts=mon[3]

        mulA=1
        if 'spe' in monA.boosts:
            temp=int(monA.boosts['spe'])
            if temp>0:
                mulA*=(2+temp)/2
            else:
                mulA*=2/(2+temp)
        mulB=1
        # print(boosts)
        if 'spe' in boosts:
            temp=int(boosts['spe'])
            if temp>0:
                mulB*=(2+temp)/2
            else:
                mulB*=2/(2+temp)
        speA=self.base_stat[name_operate.base_stat_findName(monA.species,monA.base_species,self.base_stat)][1]*mulA
        speB=self.base_stat[name_operate.base_stat_findName(mon[2][0],mon[2][1],self.base_stat)][1]*mulB
        if speA>speB:
            return 1
        return 0
    def getmoves(self,movedict,name):
        moves=list(self.movedict[m] for m in movedict)
        role=self.find_role(name,moves)
        movelist=self.get_moves_for_role(name,role)
        moves=self.move_state(movelist,moves)
        return moves
    def states(self,pokemon_data,battle,allmons):
        CurrentActive=pokemon_data[0][0]
        outS=[]
        fs=[]
        for name,mon in pokemon_data[1].items():
            if(mon[2][0]==battle.opponent_active_pokemon.species):
                for move in CurrentActive.moves.values():
                    outS.append(DamageCalculator.damage_to_state(self.DamageCalc(CurrentActive,name,mon,move,allmons),(mon[1]/100)*self.base_stat[name_operate.base_stat_findName(mon[2][0],mon[2][1],self.base_stat)][0]))
            else:
                pokemon_data[1][name][3]={}
        fs.append(outS)
        outS=[]
        for name,mon in pokemon_data[1].items():
            if(mon[2][0]!=battle.opponent_active_pokemon.species):
                for move in CurrentActive.moves.values():
                    outS.append(DamageCalculator.damage_to_state(self.DamageCalc(CurrentActive,name,mon,move,allmons),(mon[1]/100)*self.base_stat[name_operate.base_stat_findName(mon[2][0],mon[2][1],self.base_stat)][0]))
        while(len(outS)<20):
            outS.append(1)
        fs.append(outS)
        outS=[]
        for name,mon in pokemon_data[1].items():
            if(mon[2][0]==battle.opponent_active_pokemon.species):
                moves=self.getmoves(mon[0],name)
                for move in moves:
                    outS.append(DamageCalculator.damage_to_state(self.Opponent_DamageCalc(name,CurrentActive,move,allmons,mon[3]),int(CurrentActive.current_hp)))
                outS.append(self.checkSpeed(allmons,CurrentActive,mon))
                for Pokemon in battle.available_switches:
                    for move in moves:
                        outS.append(DamageCalculator.damage_to_state(self.Opponent_DamageCalc(name,Pokemon,move,allmons,mon[3]),int(Pokemon.current_hp)))
                    outS.append(self.checkSpeed(allmons,Pokemon,mon))
                while(len(outS)<30):
                    outS.append(1)
        fs.append(outS)
        outS=[]
        return fs
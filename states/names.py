from poke_env import Player, ShowdownServerConfiguration, AccountConfiguration

class name_operate:
    def getName(mons,Pokemon):
        if Pokemon.species in mons:
            return mons[Pokemon.species]
        if Pokemon.base_species in mons:
            return mons[Pokemon.base_species]
        print(Pokemon.species,Pokemon.base_species)
        return "Arceus" 

    def species(Pokemon):
        return [Pokemon.species,Pokemon.base_species]
    def convertName(ltemp,name=None):
        mon = ""
        if name!=None:
            for c in name:
                if 'a'<=c<='z':
                    mon+=name
            return mon
        for i in range(1, len(ltemp)):
            mon += ltemp[i]
        mon_standard = ""
        mon = mon.lower()
        return mon
    def searchMon(mon,pokemon_data):
        for mons, pkmn in pokemon_data[1].items():
            if mon == pkmn[2][0] or mon == pkmn[2][1]:
                return mons
        return "None"
    def operation(inp):
        inp=inp.lower()
        out=''
        for i in inp:
            if 'a'<=i<='z' or '0'<=i<='9':
                out+=i
        return out
    def base_stat_findName(name1,name2,base_stats):
        if name2 in base_stats:
            return name2
        elif name1 in base_stats:
            return name1
        print("Pokemon",name1,name2,"not in base_stats")
        return "arceus"
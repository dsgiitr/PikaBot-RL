import json
import csv
final={}

with open('poke_battle.json','r') as f:
    temp=json.load(f)
    leveldict={mon:temp[mon]['level'] for mon in temp}
# print(leveldict)
with open("Pokemon.csv",'r') as fh:
    reader=csv.reader(fh)
    for f in reader:
        if f[0]!="ID" and f[1] in leveldict:
            hp=int(((2*int(f[6])+52)*leveldict[f[1]])/100)+leveldict[f[1]]+10
            f[1]=f[1].lower()
            name=''
            for c in f[1]:
                if 'a'<=c<='z':
                    name+=c
            final[name]=[hp,int(f[11])]

# print(final)
with open("HP_Speed.json",'w') as f:
    json.dump(final,f)
# for move in moves:
#     newm=''
#     for c in move:
#         if 'a'<=c<='z':
#             newm+=c
#     final[newm]=moves[move]

# with open("itemmap1.json",'w') as fh:
#     json.dump(final,fh)
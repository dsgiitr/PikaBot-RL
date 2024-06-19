import json



#1st part

def find_role(self,pokemon_name, moves, item=None):
    if pokemon_name in self.data:
        pokemon = self.data[pokemon_name]
        for role_name, role_data in pokemon["roles"].items():
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

# Example usage


pokemon_name = "Alcremie"
move_name = ["Calm Mind","Recover"]
role=find_role(pokemon_name, move_name)
print(role)
pokemon_name = "Alcremie"
moves = get_moves_for_role(pokemon_name, role)
print(moves)
print(move_state(moves,move_name))
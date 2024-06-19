import { Generation, TypeName, StatID } from './data/interface';
export declare const SEED_BOOSTED_STAT: {
    [item: string]: StatID;
};
export declare function getItemBoostType(item: string | undefined): "Fire" | "Water" | "Grass" | "Electric" | "Ice" | "Psychic" | "Dark" | "Dragon" | "Normal" | "Fighting" | "Flying" | "Poison" | "Ground" | "Rock" | "Bug" | "Ghost" | "Steel" | "Fairy" | undefined;
export declare function getBerryResistType(berry: string | undefined): "Fire" | "Water" | "Grass" | "Electric" | "Ice" | "Psychic" | "Dark" | "Dragon" | "Normal" | "Fighting" | "Flying" | "Poison" | "Ground" | "Rock" | "Bug" | "Ghost" | "Steel" | "Fairy" | undefined;
export declare function getFlingPower(item?: string): 0 | 10 | 100 | 40 | 20 | 50 | 80 | 60 | 130 | 85 | 110 | 95 | 90 | 70 | 30;
export declare function getNaturalGift(gen: Generation, item: string): {
    t: TypeName;
    p: number;
};
export declare function getTechnoBlast(item: string): "Fire" | "Water" | "Electric" | "Ice" | undefined;
export declare function getMultiAttack(item: string): TypeName | undefined;

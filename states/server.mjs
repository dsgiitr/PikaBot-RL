import express from "express";
import { calculate, Generations, Pokemon, Move, ABILITIES } from "@smogon/calc";

const app = express();
const port = 5000;

app.get("/", (req, res) => {
  res.send("Welcome to the Pokemon damage calculator API!");
});

app.get("/calculate", (req, res) => {
  try {
    const gen = Generations.get(9); 
    const attacker = req.query.attacker || "arceusfighting";
    const defender = req.query.defender || "sandaconda";
    const move = req.query.move || "sacredfire";
    const item1 = req.query.item1 || "";
    const item2 = req.query.item2 || "";
    const nature = "Hardy";
    const level1 = parseInt(req.query.level1) || 100;
    const level2 = parseInt(req.query.level2) || 100;
    const evs = JSON.parse('{"hp": 85, "atk": 85, "def": 85, "spa": 85, "spd": 85, "spe": 85}'); // DO NOT TOUCH THESE
    const boosts1 = JSON.parse(req.query.boosts1 || '{}');
    const boosts2 = JSON.parse(req.query.boosts2 || '{}');
    const teratype1=req.query.teratype1 || "";
    const teratype2=req.query.teratype2 || "";

    const attackerPokemon = new Pokemon(gen, attacker, {
      level: level1,
      item: item1,
      nature: nature,
      evs: evs,
      boosts: boosts1,
      teraType: teratype1
       
    });

    const defenderPokemon = new Pokemon(gen, defender, {
      level: level2,
      item: item2,
      nature: nature,
      evs: evs,
      boosts: boosts2,
      teraType: teratype2
    });

    const moveObject = new Move(gen, move);

    // console.log("Attacker Pokemon:", attackerPokemon);
    // console.log("Defender Pokemon:", defenderPokemon);
    // console.log("Move:", moveObject);

    const result = calculate(gen, attackerPokemon, defenderPokemon, moveObject);

    res.status(200).send(JSON.stringify(result.damage));
  } catch (error) {
    console.error("Error during calculation:", error);
    res.status(500).send({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

app.get("/about", (req, res) => {
  res.send("About page");
});
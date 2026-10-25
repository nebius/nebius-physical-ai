"""Select one generated Go1 inspection situation for the native workflow."""

MARBLE_GO1_PRESETS: dict[str, dict[str, str]] = {
    "warehouse": {
        "workflow_preset": "warehouse",
        "world_source": "generate",
        "world_prompt": "A photorealistic modern distribution warehouse, "
        "tall pallet racks, wrapped cartons and loading "
        "bays. A continuous flat concrete central aisle at "
        "least ten meters long and three meters wide, clear "
        "space at both ends for a quadruped to turn, camera "
        "at the center of the aisle close to floor level. No "
        "people, robots, forklifts in motion, stairs or "
        "floor holes. Pallets and equipment remain outside "
        "the aisle. Realistic overhead industrial lighting.",
        "frames": "500",
        "speed_mps": "1.2",
    },
    "shipyard": {
        "workflow_preset": "shipyard",
        "world_source": "generate",
        "world_prompt": "A photorealistic dry shipyard maintenance area "
        "beside a large ship hull on supports, steel "
        "sections, cable reels and dockside maintenance "
        "equipment. A continuous dry flat concrete service "
        "lane at least ten meters long and three meters wide, "
        "clear turning areas at both ends, camera centered on "
        "the lane close to floor level. The hull, supports "
        "and equipment remain outside the lane. No people, "
        "robots, moving cranes, water on the route, stairs or "
        "floor holes. Overcast daylight.",
        "frames": "750",
        "speed_mps": "0.7",
    },
    "home": {
        "workflow_preset": "home",
        "world_source": "generate",
        "world_prompt": "A photorealistic spacious single-story accessible home "
        "with a long hallway opening into a living room, sofa, "
        "dining furniture and bookshelves. A continuous level "
        "matte wood floor along a central route at least ten "
        "meters long and two meters wide, open turning space at "
        "both ends, camera centered on the route close to floor "
        "level. All furniture stays beside the route. No people, "
        "pets, robots, rugs, loose cables, door thresholds or "
        "stairs. Natural soft window lighting.",
        "frames": "1000",
        "speed_mps": "0.35",
    },
    "factory": {
        "workflow_preset": "factory",
        "world_source": "generate",
        "world_prompt": "A photorealistic clean manufacturing workshop with "
        "rows of closed CNC machine enclosures, control "
        "cabinets and tool storage beside a marked inspection "
        "aisle. A continuous flat concrete aisle at least ten "
        "meters long and three meters wide, unobstructed "
        "turning space at both ends, camera centered on the "
        "aisle close to floor level. Enclosures and cabinets "
        "are outside the aisle. No people, robots, moving "
        "machinery, loose cables, stairs or floor holes. "
        "Neutral industrial ceiling lighting.",
        "frames": "1000",
        "speed_mps": "0.5",
    },
    "utility": {
        "workflow_preset": "utility",
        "world_source": "generate",
        "world_prompt": "A photorealistic utility pump room with stationary "
        "pump housings, pipe manifolds, valve wheels and "
        "electrical cabinets along the walls. A continuous dry "
        "level concrete service aisle at least ten meters long "
        "and three meters wide, open turning areas at both "
        "ends, camera centered on the aisle close to floor "
        "level. Pipes, valves and equipment remain outside the "
        "aisle. No people, robots, moving machinery, steps, "
        "trenches, puddles or floor holes. Realistic overhead "
        "service lighting.",
        "frames": "1000",
        "speed_mps": "0.5",
    },
}

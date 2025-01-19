#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mediator.py
@Time    :   2023/05/16 10:22:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import numpy as np
import re
import copy
from abc import ABC, abstractmethod
from pysc2.lib import units
@staticmethod
def get_minigrid_words():
    colors = ["red", "green", "blue", "yellow", "purple", "grey"]
    objects = [
        "unseen",
        "empty",
        "wall",
        "floor",
        "box",
        "key",
        "ball",
        "door",
        "goal",
        "agent",
        "lava",
    ]

    verbs = [
        "pick",
        "avoid",
        "get",
        "find",
        "put",
        "use",
        "open",
        "go",
        "fetch",
        "reach",
        "unlock",
        "traverse",
    ]

    extra_words = [
        "up",
        "the",
        "a",
        "at",
        ",",
        "square",
        "and",
        "then",
        "to",
        "of",
        "rooms",
        "near",
        "opening",
        "must",
        "you",
        "matching",
        "end",
        "hallway",
        "object",
        "from",
        "room",
    ]

    all_words = colors + objects + verbs + extra_words
    assert len(all_words) == len(set(all_words))
    return {word: i for i, word in enumerate(all_words)}

# Map of agent direction, 0: East; 1: South; 2: West; 3: North
DIRECTION = {
    0: [1, 0],
    1: [0, 1],
    2: [-1, 0],
    3: [0, -1],
}

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
} 
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

# Map of skill names to integers
SKILL_TO_IDX = {"explore": 0, "go to object": 1, "pickup": 2, "drop": 3, "toggle": 4}
IDX_TO_SKILL = dict(zip(SKILL_TO_IDX.values(), SKILL_TO_IDX.keys()))

SKILL_TO_IDX_SC2 = {"do_nothing": 0, "harvest_minerals": 1, "build_supply_depot": 2, "build_barracks": 3, "train_marine": 4, "attack": 5}
SKILL_TO_IDX_SC2_2 = {"do_nothing": 0, "harvest_minerals": 1, "build_supply_depot": 2, "build_barracks": 3, "train_marine": 4, "attack_main_base": 5, "attack_sub_base": 6, "attack_remaining_hidden_structures": 7}


class Base_Mediator(ABC):
    """The base class for Base_Mediator."""

    def __init__(self, soft):
        super().__init__()
        self.soft = soft
        self.obj_coordinate = {}

    # obs to natural language
    def RL2LLM(self, obs, color_info=True):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]
        agent_dir = agent_map[agent_pos[0],agent_pos[1]]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)

        carrying = "nothing"
        if len(key_list):
            for key in key_list:
                i, j = key
                if color_info:
                    color = obs[i,j,1]
                    obj = f"{IDX_TO_COLOR[color]} key"
                else:
                    obj = "key"

                if (agent_pos == key).all():
                    carrying = obj
                else:
                    context += f"<{obj}>, " 
                    self.obj_coordinate[obj] = (i,j)

        if len(door_list):
            for door in door_list:
                i, j = door
                if color_info:
                    color = obs[i,j,1]
                    obj = f"{IDX_TO_COLOR[color]} door"
                else:
                    obj = "door"
                
                context += f"<{obj}>, "
                self.obj_coordinate[obj] = (i,j)

        if context == '':
            context += "<nothing>, "
        context += f"holds <{carrying}>."
        
        context = f"Agent sees {context}"
        return context

    def LLM2RL(self, plans, probs):
        if self.soft:
            skill_list = [self.parser(plan) for plan in plans]
        else:
            plan = np.random.choice(plans, p=probs)
            skill_list = [self.parser(plan)]
            probs = [1.]
        
        # print("plans:", plans)
        # print("skill_list:", skill_list)
        return skill_list, probs
    
    def reset(self):
        self.obj_coordinate = {}

class SimpleDoorKey_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        return super().RL2LLM(obs, color_info=False)
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door"]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    coordinate = self.obj_coordinate["key"]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None

            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list
    

class ColoredDoorKey_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        return super().RL2LLM(obs)
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    words = text.split(' ')
                    filter_words = []
                    for w in words:
                        w1="".join(c for c in w if c.isalpha())
                        filter_words.append(w1)
                    object_word = filter_words[-2] + " " + filter_words[-1]
                    coordinate = self.obj_coordinate[object_word]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]    
                    words = text.split(' ')
                    filter_words = []
                    for w in words:
                        w1="".join(c for c in w if c.isalpha())
                        filter_words.append(w1)
                    object_word = filter_words[-2] + " " + filter_words[-1]
                    coordinate = self.obj_coordinate[object_word]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                print("Unknown Planning :", text)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
                
            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list
    
class TwoDoor_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]
        agent_dir = agent_map[agent_pos[0],agent_pos[1]]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)

        carrying = "nothing"
        if len(key_list):
            for key in key_list:
                i, j = key
                obj = "key"

                if (agent_pos == key).all():
                    carrying = obj
                else:
                    context += f"<{obj}>, " 
                    self.obj_coordinate[obj] = (i,j)

        if len(door_list):
            n = 1
            for door in door_list:
                i, j = door
                obj = f"door{n}"
                n += 1
                
                context += f"<{obj}>, "
                self.obj_coordinate[obj] = (i,j)

        if context == '':
            context += "<nothing>, "
        context += f"holds <{carrying}>."
        
        context = f"Agent sees {context}"
        return context
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "door1" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door1"]
                elif "door2" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door2"]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    coordinate = self.obj_coordinate["key"]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None

            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list
    

class StarCraft2_mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        labels = [
            "Number of command centers: ",
            # "Number of SCVs: ",
            # "Number of idle SCVs: ",
            "Number of supply depots: ",
            "Number of completed supply depots: ",
            "Number of barracks: ",
            "Number of completed barracks: ",
            "Number of marines: ",
            "Number of marines_in_production: ",
            "Free supply available: ",
            "Can afford supply depot (1 for yes, 0 for no): ",
            "Can afford barracks (1 for yes, 0 for no): ",
            "Can afford marine (1 for yes, 0 for no): ",
            # "Number of enemy command centers: ",
            # "Number of enemy SCVs: ",
            # "Number of enemy idle SCVs: ",
            # "Number of enemy supply depots: ",
            # "Number of enemy completed supply depots: ",
            # "Number of enemy barracks: ",
            # "Number of enemy completed barracks: ",
            # "Number of enemy marines: "
        ]
        text_lines = [f"{label}{value[0]}" for label, value in zip(labels, obs[0])]
        return "\n".join(text_lines)
        
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "do_noting" in text:
                act = SKILL_TO_IDX_SC2["do_nothing"]
            elif "harvest_minerals" in text:
                act = SKILL_TO_IDX_SC2["harvest_minerals"]
            elif "build_supply_depot" in text:
                act = SKILL_TO_IDX_SC2["build_supply_depot"]
            elif "build_barracks" in text:
                act = SKILL_TO_IDX_SC2["build_barracks"]
            elif "train_marine" in text:
                act = SKILL_TO_IDX_SC2["train_marine"]
            elif "attack" in text:
                act = SKILL_TO_IDX_SC2["attack"]
            else:
                act = SKILL_TO_IDX_SC2["do_nothing"]
            
            skill = {"action": act}
            skill_list.append(skill)
        
        return skill_list
    
class StarCraft2_2_mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        labels = [
            "Number of command centers: ",
            "Number of supply depots: ",
            "Number of completed supply depots: ",
            "Number of barracks: ",
            "Number of completed barracks: ",
            "Number of marines: ",
            "Number of marines_in_production: ",
            "Free supply available: ",
            "Can afford supply depot (1 for yes, 0 for no): ",
            "Can afford barracks (1 for yes, 0 for no): ",
            "Can afford marine (1 for yes, 0 for no): ",
            "main_base_facility_status (0 for No building, 1 for Unknown, 2 for Building exists): ",
            "sub_base_facility_status (0 for No building, 1 for Unknown, 2 for Building exists): ",
        ]
        text_lines = [f"{label}{value[0]}" for label, value in zip(labels, obs[0])]
        return "\n".join(text_lines)
        
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "do_noting" in text:
                act = SKILL_TO_IDX_SC2_2["do_nothing"]
            elif "harvest_minerals" in text:
                act = SKILL_TO_IDX_SC2_2["harvest_minerals"]
            elif "build_supply_depot" in text:
                act = SKILL_TO_IDX_SC2_2["build_supply_depot"]
            elif "build_barracks" in text:
                act = SKILL_TO_IDX_SC2_2["build_barracks"]
            elif "train_marine" in text:
                act = SKILL_TO_IDX_SC2_2["train_marine"]
            elif "attack_main_base" in text:
                act = SKILL_TO_IDX_SC2_2["attack_main_base"]
            elif "attack_sub_base" in text:
                act = SKILL_TO_IDX_SC2_2["attack_sub_base"]
            elif "attack_remaining_hidden_structures" in text:
                act = SKILL_TO_IDX_SC2_2["attack_remaining_hidden_structures"]
            else:
                act = SKILL_TO_IDX_SC2_2["do_nothing"]
                
            skill = {"action": act}
            skill_list.append(skill)
            
        return skill_list

if __name__ == "__main__":
    word = get_minigrid_words()
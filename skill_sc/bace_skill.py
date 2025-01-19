class BaseSkill():
    def __init__(self):
        pass
        
    def unpack_obs(self, obs):
        pass
    
# do_nothing, harvest_minerals, build_supply_depot, build_barracks, train_marine, attack
class Do_nothing(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 0, True

class Harvest_minerals(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 1, True

class Build_supply_depot(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 2, True

class Build_barracks(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 3, True

class Train_marine(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 4, True

class Attack(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 5, True
    
class Attack_main_base(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 5, True

class Attack_sub_base(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 6, True
    
class Attack_remaining_hidden_structures(BaseSkill):
    def __init__(self):
        pass
    
    def __call__(self, obs):
        return 7, True
    
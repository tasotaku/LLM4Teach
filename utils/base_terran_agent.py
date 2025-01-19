import numpy as np
import random
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units


class BaseTerranAgent(base_agent.BaseAgent):
  actions = ("do_nothing",
             "harvest_minerals", 
             "build_supply_depot", 
             "build_barracks", 
             "train_marine", 
             "attack")
  
  def get_my_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]
  
  def get_enemy_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.ENEMY]
  
  def get_my_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]
    
  def get_enemy_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.ENEMY]
    
  def get_distances(self, obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    if len(units_xy) == 0:
        return np.array([])  # 空の配列を返す
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
  
  def count_units_near_bases(self, obs, units):
    dictance_to_main_base = self.get_distances(obs, units, self.main_base_xy)
    dictance_to_sub_base = self.get_distances(obs, units, self.sub_base_xy)
    return np.sum(dictance_to_main_base < dictance_to_sub_base), np.sum(dictance_to_main_base >= dictance_to_sub_base)
  
  def count_enemy_base_facilities(self, obs):
    num_main_base_facilities = 0
    num_sub_base_facilities = 0
    command_centers = self.get_enemy_units_by_type(obs, units.Terran.CommandCenter)
    supply_depots = self.get_enemy_units_by_type(obs, units.Terran.SupplyDepot)
    barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
    
    num_main_base_facilities_tmp, num_sub_base_facilities_tmp = self.count_units_near_bases(obs, command_centers)
    num_main_base_facilities += num_main_base_facilities_tmp
    num_sub_base_facilities += num_sub_base_facilities_tmp
    num_main_base_facilities_tmp, num_sub_base_facilities_tmp = self.count_units_near_bases(obs, supply_depots)
    num_main_base_facilities += num_main_base_facilities_tmp
    num_sub_base_facilities += num_sub_base_facilities_tmp
    num_main_base_facilities_tmp, num_sub_base_facilities_tmp = self.count_units_near_bases(obs, barrackses)
    num_main_base_facilities += num_main_base_facilities_tmp
    num_sub_base_facilities += num_sub_base_facilities_tmp
    
    return num_main_base_facilities, num_sub_base_facilities
      

  def step(self, obs):
    super(BaseTerranAgent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)
      self.sub_base_xy = (16, 48) if self.base_top_left else (41, 20)
      self.main_base_xy = (38, 44) if self.base_top_left else (19, 23)
      self.main_base_last_toggle_step = 0
      self.sub_base_last_toggle_step = 0
      self.main_base_facility_status = 1
      
      self.sub_base_facility_status = 1
    num_main_base_facilities, num_sub_base_facilities = self.count_enemy_base_facilities(obs)
    distances_from_marine_to_main_base = self.get_distances(obs, self.get_my_units_by_type(obs, units.Terran.Marine), self.main_base_xy)
    if distances_from_marine_to_main_base.size == 0:
      distance_from_nearest_marine_to_main_base = 1000
    else:
      distance_from_nearest_marine_to_main_base = np.min(distances_from_marine_to_main_base)
    distances_from_marine_to_sub_base = self.get_distances(obs, self.get_my_units_by_type(obs, units.Terran.Marine), self.sub_base_xy)
    if distances_from_marine_to_sub_base.size == 0:
      distance_from_nearest_marine_to_sub_base = 1000
    else:
      distance_from_nearest_marine_to_sub_base = np.min(distances_from_marine_to_sub_base)
    if num_main_base_facilities > 0:
      next_main_base_facility_status = 2
    elif num_main_base_facilities == 0 and distance_from_nearest_marine_to_main_base < 5:
      next_main_base_facility_status = 0
    else:
      next_main_base_facility_status = self.main_base_facility_status
      
    if num_sub_base_facilities > 0:
      next_sub_base_facility_status = 2
    elif num_sub_base_facilities == 0 and distance_from_nearest_marine_to_sub_base < 5:
      next_sub_base_facility_status = 0
    else:
      next_sub_base_facility_status = self.sub_base_facility_status
    
    if self.main_base_facility_status != next_main_base_facility_status and next_main_base_facility_status == 0:
      self.main_base_last_toggle_step = obs.observation.game_loop[0]
    if self.sub_base_facility_status != next_sub_base_facility_status and next_sub_base_facility_status == 0:
      self.sub_base_last_toggle_step = obs.observation.game_loop[0]
      
    self.main_base_facility_status = next_main_base_facility_status
    self.sub_base_facility_status = next_sub_base_facility_status
    
    if obs.observation.game_loop[0] - self.main_base_last_toggle_step >= 5000 and self.main_base_last_toggle_step == 0:
      self.main_base_facility_status = 1
    if obs.observation.game_loop[0] - self.sub_base_last_toggle_step >= 5000 and self.sub_base_last_toggle_step == 0:
      self.sub_base_facility_status = 1
    

  def do_nothing(self, obs):
    return actions.RAW_FUNCTIONS.no_op()
  
  def harvest_minerals(self, obs):
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    idle_scvs = [scv for scv in scvs if scv.order_length == 0]
    if len(idle_scvs) > 0:
      mineral_patches = [unit for unit in obs.observation.raw_units
                         if unit.unit_type in [
                           units.Neutral.BattleStationMineralField,
                           units.Neutral.BattleStationMineralField750,
                           units.Neutral.LabMineralField,
                           units.Neutral.LabMineralField750,
                           units.Neutral.MineralField,
                           units.Neutral.MineralField750,
                           units.Neutral.PurifierMineralField,
                           units.Neutral.PurifierMineralField750,
                           units.Neutral.PurifierRichMineralField,
                           units.Neutral.PurifierRichMineralField750,
                           units.Neutral.RichMineralField,
                           units.Neutral.RichMineralField750
                         ]]
      scv = random.choice(idle_scvs)
      distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
      mineral_patch = mineral_patches[np.argmin(distances)] 
      return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
          "now", scv.tag, mineral_patch.tag)
    return actions.RAW_FUNCTIONS.no_op()
  
  def build_supply_depot(self, obs):
    supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(supply_depots) <= 4 and 
        obs.observation.player.minerals >= 100 and
        len(scvs) > 0):
      num_supply_depots = len(supply_depots)
      supply_depot_xy = (22 - 2 * num_supply_depots, 26) if self.base_top_left else (35 + 2 * num_supply_depots, 42)
      distances = self.get_distances(obs, scvs, supply_depot_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
          "now", scv.tag, supply_depot_xy)
    return actions.RAW_FUNCTIONS.no_op()
    
  def build_barracks(self, obs):
    completed_supply_depots = self.get_my_completed_units_by_type(
        obs, units.Terran.SupplyDepot)
    barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
    scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
    if (len(completed_supply_depots) > 0 and len(barrackses) <= 3
        and obs.observation.player.minerals >= 150 and len(scvs) > 0):
      barracks_xy = (22, 21) if self.base_top_left else (35, 45)
      num_barrackses = len(barrackses)
      barracks_xy = (23, 18 + 3*num_barrackses) if self.base_top_left else (34, 50 - 3*num_barrackses)
      distances = self.get_distances(obs, scvs, barracks_xy)
      scv = scvs[np.argmin(distances)]
      return actions.RAW_FUNCTIONS.Build_Barracks_pt(
          "now", scv.tag, barracks_xy)
    return actions.RAW_FUNCTIONS.no_op()
    
  def train_marine(self, obs):
    completed_barrackses = self.get_my_completed_units_by_type(
        obs, units.Terran.Barracks)
    free_supply = (obs.observation.player.food_cap - 
                   obs.observation.player.food_used)
    if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
        and free_supply > 0):
      barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
      if barracks.order_length < 5:
        return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
    return actions.RAW_FUNCTIONS.no_op()
  
  def attack(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
        marine_tags = [marine.tag for marine in marines]  # すべてのMarineのタグを取得

        # 攻撃対象の設定
        if self.targeting_sub_base:
            attack_xy = self.sub_base_xy
        else:
            attack_xy = self.main_base_xy

        if self.main_base_facility_status > self.sub_base_facility_status:
            attack_xy = self.main_base_xy
        elif self.main_base_facility_status < self.sub_base_facility_status:
            attack_xy = self.sub_base_xy
        elif self.main_base_facility_status == 0 and self.sub_base_facility_status == 0:
            attack_xy = self.main_base_xy if obs.observation.game_loop[0] % 2000 < 1000 else self.sub_base_xy

        # 攻撃命令をランダムな偏差を付与して送る
        x_offset = random.randint(-6, 6)
        y_offset = random.randint(-6, 6)
        return actions.RAW_FUNCTIONS.Attack_pt(
            "now", marine_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

    return actions.RAW_FUNCTIONS.no_op()
  
  def attack_main_base(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
        marine_tags = [marine.tag for marine in marines]  # すべてのMarineのタグを取得
        attack_xy = self.main_base_xy

        # 攻撃命令をランダムな偏差を付与して送る
        x_offset = random.randint(-6, 6)
        y_offset = random.randint(-6, 6)
        return actions.RAW_FUNCTIONS.Attack_pt(
            "now", marine_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

    return actions.RAW_FUNCTIONS.no_op()
  
  def attack_sub_base(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
        marine_tags = [marine.tag for marine in marines]  # すべてのMarineのタグを取得
        attack_xy = self.sub_base_xy

        # 攻撃命令をランダムな偏差を付与して送る
        x_offset = random.randint(-6, 6)
        y_offset = random.randint(-6, 6)
        return actions.RAW_FUNCTIONS.Attack_pt(
            "now", marine_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

    return actions.RAW_FUNCTIONS.no_op()
  
  def attack_remaining_hidden_structures(self, obs):
    marines = self.get_my_units_by_type(obs, units.Terran.Marine)
    if len(marines) > 0:
        marine_tags = [marine.tag for marine in marines]  # すべてのMarineのタグを取得
        attack_xy = self.main_base_xy if obs.observation.game_loop[0] % 2000 < 1000 else self.sub_base_xy
        
        x_offset = random.randint(-6, 6)
        y_offset = random.randint(-6, 6)
        
        return actions.RAW_FUNCTIONS.Attack_pt(
            "now", marine_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        
    return actions.RAW_FUNCTIONS.no_op()
        
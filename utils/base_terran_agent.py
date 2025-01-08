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
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

  def step(self, obs):
    super(BaseTerranAgent, self).step(obs)
    if obs.first():
      command_center = self.get_my_units_by_type(
          obs, units.Terran.CommandCenter)[0]
      self.base_top_left = (command_center.x < 32)
      self.sub_base_xy = (16, 48) if self.base_top_left else (41, 20)
      self.main_base_xy = (38, 44) if self.base_top_left else (19, 23)
      self.targeting_sub_base = True
      self.last_toggle_step = 0

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
    if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
        len(scvs) > 0):
      supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
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
    if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and 
        obs.observation.player.minerals >= 150 and len(scvs) > 0):
      barracks_xy = (22, 21) if self.base_top_left else (35, 45)
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

        # サブ拠点の距離と状態の確認
        if self.targeting_sub_base:
            distances_to_sub = self.get_distances(obs, marines, self.sub_base_xy)
            distance_to_sub = np.min(distances_to_sub)

            enemy_command_centers = self.get_enemy_units_by_type(
                obs, units.Terran.CommandCenter)
            enemy_supply_depots = self.get_enemy_units_by_type(
                obs, units.Terran.SupplyDepot)
            enemy_barrackses = self.get_enemy_units_by_type(
                obs, units.Terran.Barracks)

            if distance_to_sub < 5 and len(enemy_command_centers) == 0 and len(enemy_supply_depots) == 0 and len(enemy_barrackses) == 0:
                self.targeting_sub_base = False
                self.last_toggle_step = obs.observation.game_loop[0]  # サブ拠点を攻略した時点でタイミングを記録

        # ターゲットの切り替えロジック
        current_step = obs.observation.game_loop[0]
        if current_step - self.last_toggle_step > 5000 and self.last_toggle_step != 0:  # 8000ステップ（ゲーム内時間）ごとに切り替え
            self.targeting_sub_base = not self.targeting_sub_base
            self.last_toggle_step = current_step

        # 攻撃命令をランダムな偏差を付与して送る
        x_offset = random.randint(-6, 6)
        y_offset = random.randint(-6, 6)
        return actions.RAW_FUNCTIONS.Attack_pt(
            "now", marine_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

    return actions.RAW_FUNCTIONS.no_op()
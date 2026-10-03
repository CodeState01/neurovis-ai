import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from student.virtual_env import World
class VirtualWorldTests(unittest.TestCase):
 def test_moves_stop_at_walls_and_collect_items(self):
  world=World.from_grid(['#####','#S*.#','#..G#','#####','#####']);world.move('direita');self.assertEqual(world.collected,1);self.assertEqual(world.x,2);world.move('direita');self.assertEqual(world.x,3);world.move('baixo');self.assertEqual(world.message,'Objetivo alcançado!')
 def test_rejects_invalid_map_or_direction(self):
  with self.assertRaises(ValueError):World.from_grid(['#####','#SS.#','#..G#','#####','#####'])
  with self.assertRaises(ValueError):World.from_grid(['#####','#S..#','#...#','#...#','#####'])
  world=World.from_grid(['#####','#S..#','#...#','#..G#','#####'])
  with self.assertRaises(ValueError):world.move('__import__')
if __name__=='__main__':unittest.main()

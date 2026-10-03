"""A tiny deterministic grid world for the teacher's agent plans.

The model selects from finite directions; it never writes or executes Python.
"""
from __future__ import annotations
from dataclasses import dataclass

DIRECTIONS = {'cima': (0, -1), 'baixo': (0, 1), 'esquerda': (-1, 0), 'direita': (1, 0)}

@dataclass
class World:
    grid: list[str]
    x: int
    y: int
    goal: tuple[int, int]
    collected: int = 0
    steps: int = 0
    message: str = ''

    @classmethod
    def from_grid(cls, rows):
        if not isinstance(rows, list) or not 5 <= len(rows) <= 14 or any(not isinstance(r, str) for r in rows):
            raise ValueError('O mapa precisa ter de 5 a 14 linhas.')
        width = len(rows[0])
        if not 5 <= width <= 14 or any(len(r) != width or set(r) - set('#.SG*') for r in rows):
            raise ValueError('O mapa deve ser retangular e usar só # . S G *.')
        starts = [(x,y) for y,r in enumerate(rows) for x,v in enumerate(r) if v == 'S']
        goals = [(x,y) for y,r in enumerate(rows) for x,v in enumerate(r) if v == 'G']
        if len(starts) != 1 or len(goals) != 1 or sum(r.count('*') for r in rows) > 6:
            raise ValueError('O mapa precisa ter um início, uma saída e no máximo seis itens.')
        return cls(list(rows), *starts[0], goals[0])

    def observe(self):
        return {'position': [self.x,self.y], 'goal': list(self.goal), 'itemsRemaining': sum(r.count('*') for r in self.grid), 'steps': self.steps}

    def move(self, direction):
        if direction not in DIRECTIONS: raise ValueError('Direção não permitida.')
        dx,dy=DIRECTIONS[direction];nx,ny=self.x+dx,self.y+dy;self.steps+=1
        if not (0<=ny<len(self.grid) and 0<=nx<len(self.grid[0])) or self.grid[ny][nx]=='#': self.message='Parede: movimento bloqueado.';return self.observe()
        row=list(self.grid[ny]);cell=row[nx]
        if cell=='*': self.collected+=1
        if cell!='G': row[nx]='.'
        self.grid[ny]= ''.join(row);self.x,self.y=nx,ny
        self.message='Objetivo alcançado!' if (nx,ny)==self.goal else ('Item coletado!' if cell=='*' else 'Movimento concluído.')
        return self.observe()

    def view(self):
        out=[]
        for y,row in enumerate(self.grid):
            line=list(row)
            if (self.x,self.y)!=(self.goal):line[self.x]='A'
            out.append(''.join(line))
        return out

def plan_for(task: str, size=10):
    """Use Qwen to create a constrained environment/map/action list, not code."""
    import json
    import urllib.request
    size=max(7,min(14,int(size)))
    prompt=('Crie um roteiro de movimentos para um robô numa grade segura. Não escreva código. '
      'Mapa retangular com paredes # no contorno, S no começo e G na saída. '
      'Responda JSON válido com somente os campos "grid" (linhas de mapa usando só # . S G *) e '
      '"actions" (lista de 1 a 60 direções em português: cima, baixo, esquerda, direita). '
      f'Use grade de {size}x{size} sem itens e encontre um caminho válido sem atravessar paredes. Tarefa: {task[:600]}')
    data=json.dumps({'model':'qwen3.5:4b','messages':[{'role':'user','content':prompt}],'format':'json','think':False,'stream':False,'keep_alive':'1m','options':{'temperature':0.1,'num_ctx':2048,'num_predict':1000}}).encode()
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response: raw=json.load(response).get('message',{}).get('content','')
    try: obj=json.loads(raw)
    except json.JSONDecodeError as err: raise ValueError('O professor não concluiu o plano. Tente novamente.') from err
    world=World.from_grid(obj.get('grid'))
    initial=world.view()
    actions=obj.get('actions')
    if not isinstance(actions,list) or not 1<=len(actions)<=60 or any(a not in DIRECTIONS for a in actions):raise ValueError('Plano de ações inválido; só aceitamos até 60 movimentos conhecidos.')
    history=[]
    for action in actions:
        if (world.x,world.y)==world.goal:break
        world.move(action);history.append({'action':action,'view':world.view(),'message':world.message,'observation':world.observe()})
    return {'grid':world.view(),'start':initial,'actions':actions[:len(history)],'history':history,'position':[world.x,world.y], 'goal':list(world.goal),'steps':world.steps,'collected':world.collected,'success':(world.x,world.y)==world.goal,'task':task[:600], 'description':'O Qwen planejou os movimentos; o aplicativo executou cada passo num simulador local. Nenhum código do modelo foi executado.'}

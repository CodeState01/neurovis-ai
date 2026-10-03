"""Native Python desktop application for Neurovis AI."""
from __future__ import annotations
import base64, json, os, queue, secrets, socket, subprocess, sys, threading, time, urllib.error, urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
BASE = '#10131c'; PANEL = '#191e2b'; CARD = '#212838'; TEXT = '#f1f4ff'; MUTED = '#a7b0c5'; ACCENT = '#98a8ff'; GREEN = '#68dbb4'

class NeurovisApp:
    def __init__(self, root):
        self.root = root; self.root.title('Neurovis AI · laboratório local'); self.root.geometry('1240x820'); self.root.minsize(960, 660); self.root.configure(bg=BASE)
        self.token = secrets.token_urlsafe(32); sock = socket.socket(); sock.bind(('127.0.0.1',0)); self.port = sock.getsockname()[1]; sock.close()
        self.url = f'http://127.0.0.1:{self.port}'; self.queue = queue.Queue(); self.messages=[]; self.attachment=None; self.last_snapshot=None; self.closing=False; self.loading=False; self.busy=False
        log_path=ROOT/'.runtime'/'student'/'desktop-service.log'; log_path.parent.mkdir(parents=True,exist_ok=True)
        self.log_file=log_path.open('a',encoding='utf-8'); env=os.environ.copy();env.update(STUDENT_TOKEN=self.token,STUDENT_PORT=str(self.port),PYTHONUNBUFFERED='1')
        self.service=subprocess.Popen([str(PYTHON),'-u',str(ROOT/'student'/'service.py')],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=self.log_file,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self._styles(); self._build(); self.root.protocol('WM_DELETE_WINDOW',self.close); self.root.after(80,self._drain); self.root.after(300,self._poll); self.root.after(1200,self._warm)

    def _styles(self):
        style=ttk.Style();style.theme_use('clam');style.configure('.',background=BASE,foreground=TEXT,font=('Segoe UI',10));style.configure('TFrame',background=BASE);style.configure('Panel.TFrame',background=PANEL);style.configure('TLabel',background=BASE,foreground=TEXT);style.configure('Muted.TLabel',foreground=MUTED);style.configure('Title.TLabel',font=('Segoe UI Semibold',22),foreground=TEXT);style.configure('Header.TLabel',font=('Segoe UI Semibold',12));style.configure('TNotebook',background=BASE,borderwidth=0);style.configure('TNotebook.Tab',background=PANEL,foreground=MUTED,padding=(16,10));style.map('TNotebook.Tab',background=[('selected',CARD)],foreground=[('selected',TEXT)]);style.configure('TButton',background=CARD,foreground=TEXT,borderwidth=0,padding=(12,9));style.map('TButton',background=[('active','#323d55')]);style.configure('Accent.TButton',background='#5869d8',foreground='white');style.map('Accent.TButton',background=[('active','#7180ef')]);style.configure('TEntry',fieldbackground=CARD,foreground=TEXT,insertcolor=TEXT,padding=9);style.configure('Horizontal.TProgressbar',background=ACCENT,troughcolor=CARD)

    def _build(self):
        top=ttk.Frame(self.root,padding=(22,15));top.pack(fill='x');ttk.Label(top,text='NEUROVIS',foreground=ACCENT,font=('Segoe UI Semibold',10)).pack(anchor='w');ttk.Label(top,text='Laboratório de IA local',style='Title.TLabel').pack(anchor='w',pady=(2,0));self.status=ttk.Label(top,text='Iniciando o motor Python…',style='Muted.TLabel');self.status.pack(anchor='w',pady=(3,0))
        body=ttk.Frame(self.root,padding=(18,4,18,18));body.pack(fill='both',expand=True);body.columnconfigure(0,weight=3);body.columnconfigure(1,weight=2);body.rowconfigure(0,weight=1)
        left=ttk.Frame(body,style='Panel.TFrame',padding=14);left.grid(row=0,column=0,sticky='nsew',padx=(0,10));left.rowconfigure(1,weight=1);left.columnconfigure(0,weight=1)
        bar=ttk.Frame(left,style='Panel.TFrame');bar.grid(row=0,column=0,sticky='ew');ttk.Label(bar,text='Seu aluno',style='Header.TLabel').pack(side='left');ttk.Label(bar,text='TinyLlama 1.1B · responde localmente',style='Muted.TLabel').pack(side='right')
        self.chat=tk.Text(left,wrap='word',bg=PANEL,fg=TEXT,insertbackground=TEXT,relief='flat',padx=10,pady=10,font=('Segoe UI',11),state='disabled');self.chat.grid(row=1,column=0,sticky='nsew',pady=10);self.chat.tag_configure('you',foreground=ACCENT,spacing1=9);self.chat.tag_configure('ai',foreground=GREEN,spacing1=9);self.chat.tag_configure('info',foreground=MUTED,spacing1=4);self.chat.tag_configure('error',foreground='#ff9b9b',spacing1=4)
        row=ttk.Frame(left,style='Panel.TFrame');row.grid(row=2,column=0,sticky='ew');row.columnconfigure(0,weight=1);self.entry=ttk.Entry(row);self.entry.grid(row=0,column=0,sticky='ew',padx=(0,7));self.entry.bind('<Return>',lambda _e:self.send());self.send_btn=ttk.Button(row,text='Enviar',style='Accent.TButton',command=self.send);self.send_btn.grid(row=0,column=1);ttk.Button(row,text='＋ Imagem',command=self.pick_image).grid(row=0,column=2,padx=(7,0));self.attach_label=ttk.Label(left,text='',style='Muted.TLabel');self.attach_label.grid(row=3,column=0,sticky='w',pady=(5,0))
        right=ttk.Frame(body,style='Panel.TFrame',padding=14);right.grid(row=0,column=1,sticky='nsew');right.rowconfigure(2,weight=1);right.columnconfigure(0,weight=1);ttk.Label(right,text='Professor Qwen',style='Header.TLabel').grid(row=0,column=0,sticky='w');ttk.Label(right,text='Ensine um assunto; o aluno aprende e passa a responder.',style='Muted.TLabel',wraplength=420).grid(row=1,column=0,sticky='w',pady=(5,8))
        self.tabs=ttk.Notebook(right);self.tabs.grid(row=2,column=0,sticky='nsew');learn=ttk.Frame(self.tabs,style='Panel.TFrame',padding=9);envtab=ttk.Frame(self.tabs,style='Panel.TFrame',padding=9);self.tabs.add(learn,text='Professor e neurônios');self.tabs.add(envtab,text='Ambiente virtual');learn.rowconfigure(3,weight=1);learn.columnconfigure(0,weight=1);self.goal=tk.Text(learn,height=3,wrap='word',bg=CARD,fg=TEXT,insertbackground=TEXT,relief='flat',padx=8,pady=8,font=('Segoe UI',10));self.goal.grid(row=0,column=0,sticky='ew');actions=ttk.Frame(learn,style='Panel.TFrame');actions.grid(row=1,column=0,sticky='ew',pady=8);ttk.Label(actions,text='Passos:').pack(side='left');self.steps=tk.StringVar(value='40');ttk.Spinbox(actions,from_=1,to=80,textvariable=self.steps,width=5).pack(side='left',padx=7);self.grow=tk.BooleanVar(value=True);ttk.Checkbutton(actions,text='Qwen pode criar neurônios (até 5.632)',variable=self.grow).pack(side='left',padx=4);self.teach_btn=ttk.Button(actions,text='Ensinar',style='Accent.TButton',command=self.teach);self.teach_btn.pack(side='right');ttk.Label(learn,text='Neurônios e conexões ativados agora',style='Header.TLabel').grid(row=2,column=0,sticky='w',pady=(5,3));self.canvas=tk.Canvas(learn,bg='#121723',highlightthickness=0,height=215);self.canvas.grid(row=3,column=0,sticky='nsew');self.metrics=ttk.Label(learn,text='Aguardando ativações reais do aluno.',style='Muted.TLabel',wraplength=440);self.metrics.grid(row=4,column=0,sticky='w',pady=(8,2));self.events=tk.Text(learn,height=5,wrap='word',bg=PANEL,fg=MUTED,relief='flat',padx=8,pady=7,font=('Segoe UI',9),state='disabled');self.events.grid(row=5,column=0,sticky='ew',pady=(8,0))
        envtab.columnconfigure(0,weight=1);envtab.rowconfigure(3,weight=1);ttk.Label(envtab,text='Dê uma tarefa ao robô',style='Header.TLabel').grid(row=0,column=0,sticky='w');self.env_task=ttk.Entry(envtab);self.env_task.insert(0,'Encontre o caminho até a saída.');self.env_task.grid(row=1,column=0,sticky='ew',pady=(6,4));envrow=ttk.Frame(envtab,style='Panel.TFrame');envrow.grid(row=2,column=0,sticky='ew',pady=5);ttk.Label(envrow,text='Tabuleiro:').pack(side='left');self.env_size=tk.StringVar(value='10');ttk.Combobox(envrow,textvariable=self.env_size,values=('7','8','10','12','14'),width=5,state='readonly').pack(side='left',padx=8);self.env_button=ttk.Button(envrow,text='Pedir plano ao Qwen',style='Accent.TButton',command=self.environment);self.env_button.pack(side='right');self.world_canvas=tk.Canvas(envtab,bg='#121723',highlightthickness=0,height=260);self.world_canvas.grid(row=3,column=0,sticky='nsew',pady=8);self.world_info=ttk.Label(envtab,text='O Qwen escolhe ações e o app simula cada passo num tabuleiro local.',style='Muted.TLabel',wraplength=430);self.world_info.grid(row=4,column=0,sticky='w');ttk.Label(right,text='Visão local: anexe uma imagem e pergunte. O Qwen observa; o aluno responde.',style='Muted.TLabel',wraplength=430).grid(row=3,column=0,sticky='w',pady=(9,0))

    def _request(self,path,payload=None,timeout=90):
        data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode();req=urllib.request.Request(self.url+path,data=data,headers={'X-Student-Token':self.token,'Content-Type':'application/json'});return urllib.request.urlopen(req,timeout=timeout)
    def _note(self,text,tag='info'):
        self.chat.configure(state='normal');self.chat.insert('end',text+'\n',tag);self.chat.configure(state='disabled');self.chat.see('end')
    def _warm(self):
        if self.closing:return
        def work():
            try:
                data=json.load(self._request('/status',timeout=20))
                if not data.get('loaded') and not self.loading:
                    self.loading=True;self.queue.put(('warm_start',None))
                    # Loading is triggered through a chat-safe short warm-up question, not shown in chat.
                    try:
                        response=self._request('/chat',{'messages':[{'role':'user','content':'Responda somente: pronto.'}]},timeout=240)
                        for line in response:
                            item=json.loads(line)
                            if item.get('type')=='error':raise RuntimeError(item['error'])
                    finally:self.loading=False
            except Exception as exc:self.loading=False;self.queue.put(('note',(f'Motor local: {exc}','error')))
        threading.Thread(target=work,daemon=True).start()

    def send(self):
        text=self.entry.get().strip()
        if not text:return
        if self.busy:return
        self.busy=True;attach=self.attachment;self.entry.delete(0,'end');self.attachment=None;self.attach_label.configure(text='');self._note('Você: '+text,'you');self._note('Aluno: pensando…','ai');self.send_btn.configure(state='disabled')
        def work():
            try:
                messages=self.messages[-10:]+[{'role':'user','content':text}]
                if attach:
                    image={k:v for k,v in attach.items() if k in ('image','mime')}
                    vision=json.load(self._request('/vision',{'question':text,**image},timeout=300));messages[-1]['content']+='\n\nDescrição visual feita pelo Qwen local: '+vision['description'];self.queue.put(('note',('Qwen observou a imagem. O aluno está preparando a resposta…','info')))
                req=self._request('/chat',{'messages':messages},timeout=300);answer='';
                for raw in req:
                    event=json.loads(raw)
                    if event['type']=='token':answer+=event.get('content','')
                    elif event['type']=='error':raise RuntimeError(event.get('error','Falha ao gerar resposta.'))
                self.messages=messages+[{'role':'assistant','content':answer}];self.queue.put(('answer',answer or 'Não consegui gerar uma resposta. Tente novamente.'))
            except urllib.error.HTTPError as exc:
                try:err=json.load(exc).get('error',str(exc))
                except Exception:err=str(exc)
                self.queue.put(('error',err))
            except Exception as exc:self.queue.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def pick_image(self):
        path=filedialog.askopenfilename(title='Escolher imagem para o Qwen observar',filetypes=[('Imagens','*.png *.jpg *.jpeg *.webp *.bmp'),('Todos os arquivos','*.*')])
        if not path:return
        try:
            from PIL import Image,ImageOps
            from io import BytesIO
            with Image.open(path) as im:
                if im.width>4096 or im.height>4096:raise ValueError('A imagem deve ter no máximo 4096 × 4096 pixels.')
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1280,1280));b=BytesIO();im.save(b,format='JPEG',quality=84,optimize=True);self.attachment={'image':base64.b64encode(b.getvalue()).decode(),'mime':'image/jpeg','name':Path(path).name}
            self.attach_label.configure(text=f'Imagem anexada: {Path(path).name} — envie sua pergunta para o aluno.')
        except Exception as exc:messagebox.showerror('Não foi possível abrir a imagem',str(exc))

    def teach(self):
        goal=self.goal.get('1.0','end').strip()
        if len(goal)<3:messagebox.showinfo('Tema de estudo','Escreva um tema para o professor.');return
        try:steps=int(self.steps.get())
        except ValueError:steps=40
        self.teach_btn.configure(state='disabled');self._note('Pedido enviado ao professor Qwen…','info')
        def work():
            try:json.load(self._request('/teach',{'goal':goal,'steps':steps,'allowGrowth':self.grow.get()},timeout=30));self.queue.put(('note',('Qwen começou a planejar a aula; acompanhe o progresso no painel.','info')))
            except Exception as e:self.queue.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()

    def environment(self):
        task=self.env_task.get().strip()
        if len(task)<3:messagebox.showinfo('Tarefa','Descreva o que deseja que o robô faça.');return
        self.env_button.configure(state='disabled');self.world_info.configure(text='Qwen planejando…')
        def work():
            try:self.queue.put(('world',json.load(self._request('/environment',{'task':task,'size':int(self.env_size.get())},timeout=240))))
            except Exception as e:self.queue.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()

    def _draw_world(self,result):
        c=self.world_canvas;c.delete('all');grid=result.get('start',[]);hist=result.get('history',[]);w=max(200,c.winfo_width());h=max(170,c.winfo_height());rows=len(grid) or 1;cols=max(map(len,grid)) if grid else 1;cell=min((w-20)/cols,(h-20)/rows);ox=(w-cell*cols)/2;oy=(h-cell*rows)/2
        view=grid
        for step in hist:
            view=step['view'];c.delete('all')
            for y,line in enumerate(view):
                for x,ch in enumerate(line):
                    color={'#':'#41495d','.':'#252d3e','S':'#445574','G':'#48ad85','*':'#cda74a','A':'#667af3'}.get(ch,'#252d3e');c.create_rectangle(ox+x*cell,oy+y*cell,ox+(x+1)*cell-2,oy+(y+1)*cell-2,fill=color,outline='#10131c')
                    if ch in ('A','G','*'):c.create_text(ox+(x+.5)*cell,oy+(y+.5)*cell,text={'A':'●','G':'★','*':'◆'}.get(ch,ch),fill='white',font=('Segoe UI',max(8,int(cell*.42))))
            c.update();time.sleep(.035)
        self.world_info.configure(text=f"{'Saída alcançada' if result.get('success') else 'Saída não alcançada'} · {result.get('steps',0)} passos · {result.get('collected',0)} itens. {result.get('description','')}");self.env_button.configure(state='normal')

    def _poll(self):
        if self.closing:return
        def work():
            try:self.queue.put(('state',json.load(self._request('/status',timeout=8))))
            except Exception as e:self.queue.put(('state_error',str(e)))
        threading.Thread(target=work,daemon=True).start();self.root.after(1600,self._poll)
    def _drain(self):
        try:
            while True:
                kind,val=self.queue.get_nowait()
                if kind=='note':self._note(*val)
                elif kind=='warm_start':self.send_btn.configure(state='disabled');self._note('Preparando o aluno pela primeira vez. Aguarde um instante…','info')
                elif kind=='answer':self.busy=False;self._note('Aluno: '+val,'ai');self.send_btn.configure(state='normal')
                elif kind=='error':self.busy=False;self._note('Erro: '+val,'error');self.send_btn.configure(state='normal');self.teach_btn.configure(state='normal');self.env_button.configure(state='normal')
                elif kind=='state_error':self.status.configure(text='Reconectando ao motor Python…')
                elif kind=='state':self._update_state(val)
                elif kind=='world':self._draw_world(val);self.env_button.configure(state='normal')
        except queue.Empty:pass
        if not self.closing:self.root.after(80,self._drain)
    def _update_state(self,s):
        phase=s.get('phase','idle');labels={'idle':'Pronto','answering':'Aluno respondendo…','training':'Aluno treinando…','teacher':'Qwen preparando a aula…','loading':'Trocando professor e aluno…','vision':'Qwen observando a imagem local…','environment':'Qwen planejando ações locais…'};self.status.configure(text=f"{labels.get(phase,phase)}  ·  {s.get('device','GPU/CPU local')}");self.send_btn.configure(state='disabled' if s.get('busy') or self.loading else 'normal');self.teach_btn.configure(state='disabled' if s.get('busy') else 'normal');info=s.get('message','');self._event(info);snap=s.get('snapshot')
        if snap and snap.get('available') and snap!=self.last_snapshot:self.last_snapshot=snap;self._draw(snap)
        if s.get('metrics'):
            m=s['metrics'];self.metrics.configure(text=f"Validação antes: {m.get('validationBefore',0):.3f}  ·  depois: {m.get('validationAfter',0):.3f}  ·  neurônios: {s.get('addedNeurons',0):,}".replace(',','.'))
    def _event(self,text):
        if not text:return
        self.events.configure(state='normal');last=self.events.get('1.0','end').strip().splitlines()[-1:] if self.events.get('1.0','end').strip() else []
        if not last or last[0]!=text:self.events.insert('end',text+'\n');self.events.delete('1.0','end-6c') if int(self.events.index('end-1c').split('.')[0])>6 else None
        self.events.configure(state='disabled');self.events.see('end')
    def _draw(self,s):
        c=self.canvas;c.delete('all');w=max(200,c.winfo_width());h=max(120,c.winfo_height());layers=s.get('activations',[]);weights=s.get('weights',[])
        if not layers or not weights:c.create_text(w/2,h/2,text='Envie uma mensagem para ver ativações do aluno',fill=MUTED);return
        points=[]
        for li,vals in enumerate(layers):
            x=35+(w-70)*li/max(1,len(layers)-1);ys=[(h-30)*(i+1)/(len(vals)+1)+8 for i in range(len(vals))];points.append([(x,y) for y in ys])
        for li,mat in enumerate(weights):
            for oi,row in enumerate(mat):
                for ii,weight in enumerate(row):
                    if abs(weight)<.06:continue
                    x1,y1=points[li][ii];x2,y2=points[li+1][oi];c.create_line(x1,y1,x2,y2,fill=GREEN if weight>0 else '#fa8e9a',width=min(2,abs(weight)*4),stipple='gray50')
        for li,vals in enumerate(layers):
            for i,value in enumerate(vals):
                x,y=points[li][i];active=min(1,abs(value));color='#%02x%02x%02x'%(int(35+60*active),int(49+145*active),int(75+90*active));c.create_oval(x-5,y-5,x+5,y+5,fill=color,outline=ACCENT,width=1);c.create_text(x,y-12,text=f'{value:.1f}',fill=MUTED,font=('Segoe UI',7))
        for li,label in enumerate(('Entrada','Neurônios','Saída')):c.create_text(points[min(li,len(points)-1)][0][0],h-12,text=label,fill=MUTED,font=('Segoe UI',8))
    def close(self):
        self.closing=True
        try:
            req=urllib.request.Request(self.url+'/stop',data=b'{}',headers={'X-Student-Token':self.token,'Content-Type':'application/json'});urllib.request.urlopen(req,timeout=1).close()
        except Exception:pass
        try:self.service.stdin.close();self.service.wait(timeout=3)
        except Exception:self.service.terminate()
        try:self.log_file.close()
        except Exception:pass
        self.root.destroy()

if __name__=='__main__':
    if not PYTHON.exists():
        print('Execute Instalar-Neurovis.cmd primeiro para preparar o Python e os modelos locais.');sys.exit(1)
    root=tk.Tk();app=NeurovisApp(root);root.mainloop()

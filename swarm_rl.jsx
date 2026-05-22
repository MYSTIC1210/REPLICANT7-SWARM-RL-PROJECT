import { useState, useEffect, useRef, useCallback } from "react";

/*
╔══════════════════════════════════════════════════════════════════════════════╗
║  REPLICANT-7  —  Multi-Agent Reinforcement Learning Swarm                  ║
║  Saronic-style Autonomous Maritime Drone Coordination                       ║
║  MADDPG · Sparse Comms · Emergent Behaviour · Live Policy Visualiser       ║
╚══════════════════════════════════════════════════════════════════════════════╝
*/

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const CFG = {
  W: 720, H: 680,
  N_AGENTS: 8,
  N_ZONES: 3,           // patrol / deny / recon zones
  N_THREATS: 4,
  COMM_RANGE: 160,      // acoustic link threshold
  COMM_BUDGET: 3,       // max msgs per agent per tick (bandwidth constraint)
  SENSOR_R: 90,
  SEP_DIST: 48,
  TRAIL: 55,
  TICK_MS: 52,
  // RL hyper-params (displayed, simulated)
  GAMMA: 0.95,
  LR_ACTOR: 3e-4,
  LR_CRITIC: 1e-3,
  EPSILON_START: 1.0,
  EPSILON_END: 0.05,
  EPSILON_DECAY: 0.9995,
  REWARD_COVERAGE: 1.2,
  REWARD_COMM_PENALTY: -0.4,   // penalises over-broadcasting
  REWARD_COLLISION: -2.0,
  REWARD_ZONE_CLEAR: 3.0,
};

// ─── DESIGN ───────────────────────────────────────────────────────────────────
const D = {
  bg0:"#0b0e14", bg1:"#0d1219", bg2:"#111720", bg3:"#141d28",
  border:"#1a2d42", borderHi:"#2a4d72",
  r:"#ff3355",   rD:"rgba(255,51,85,.14)",
  g:"#00f5a0",   gD:"rgba(0,245,160,.12)",
  b:"#00b4ff",   bD:"rgba(0,180,255,.12)",
  y:"#ffcc00",   yD:"rgba(255,204,0,.12)",
  o:"#ff7733",   oD:"rgba(255,119,51,.12)",
  p:"#cc44ff",   pD:"rgba(204,68,255,.12)",
  muted:"#0d1e2e", mutedHi:"#162a3e",
  text:"#7ab8d8", textHi:"#b8ddf0", textDim:"#2a4a65", textXd:"#162030",
  grid:"#090d12",
  FH:"'Rajdhani',sans-serif",
  FM:"'Share Tech Mono','Courier New',monospace",
  FD:"'Orbitron',monospace",
};

// agent colour by id
const AC = [D.b,D.g,D.y,D.o,D.r,D.p,"#00ffee","#ff88cc"];

// ─── MATH ────────────────────────────────────────────────────────────────────
const rnd   = (a,b)=>Math.random()*(b-a)+a;
const rndI  = (a,b)=>Math.floor(rnd(a,b));
const d2    = (a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const clamp = (v,a,b)=>Math.max(a,Math.min(b,v));
const lerp  = (a,b,t)=>a+(b-a)*t;
const norm  = (vx,vy)=>{const m=Math.hypot(vx,vy)||1;return[vx/m,vy/m];};

// Simplex-like noise for current field
const noise=(x,y,t)=>{
  const s=0.007, ts=0.0025;
  return{
    x:Math.sin(y*s+t*ts)*0.045+Math.cos(x*s*1.3+t*ts*0.7)*0.022,
    y:Math.cos(x*s+t*ts)*0.045+Math.sin(y*s*1.3+t*ts*0.7)*0.022,
  };
};

// ─── SIMULATED RL POLICY (tabular Q approximation for display) ───────────────
// In a real system this would be a neural network. Here we simulate the
// learning curve and emergent communication strategy numerically.
class PolicySim {
  constructor(id){
    this.id=id;
    this.epsilon=CFG.EPSILON_START;
    this.episode=0;
    this.qTable={};           // state→action value estimates
    this.lastReward=0;
    this.cumReward=0;
    this.rewardHistory=[];
    this.commSent=0;
    this.exploitRate=0;       // fraction greedy decisions
    this.actorLoss=rnd(2,4);
    this.criticLoss=rnd(3,6);
  }
  update(reward){
    this.lastReward=reward;
    this.cumReward+=reward;
    this.epsilon=Math.max(CFG.EPSILON_END, this.epsilon*CFG.EPSILON_DECAY);
    this.episode++;
    this.exploitRate=1-this.epsilon;
    this.actorLoss=Math.max(0.01,this.actorLoss*0.9992+rnd(-0.05,0.03));
    this.criticLoss=Math.max(0.02,this.criticLoss*0.9990+rnd(-0.08,0.04));
    this.rewardHistory=[...this.rewardHistory,this.cumReward].slice(-60);
  }
}

// ─── STATE FACTORIES ─────────────────────────────────────────────────────────
function makeAgent(id){
  const ang=(id/CFG.N_AGENTS)*Math.PI*2+rnd(-.4,.4);
  const r=rnd(80,160);
  return{
    id, col:AC[id],
    x:CFG.W/2+Math.cos(ang)*r, y:CFG.H/2+Math.sin(ang)*r,
    vx:rnd(-1,1), vy:rnd(-1,1),
    policy:new PolicySim(id),
    status:"patrol",      // patrol|intercept|cover|RTB|idle
    trail:[],
    commBudget:CFG.COMM_BUDGET,
    msgOut:[], msgIn:[],
    coverageScore:0,
    zoneAssigned:id%CFG.N_ZONES,
    sensorActive:true,
    energy:rnd(85,100),
    collisions:0,
    battery:rnd(82,100),
    // emergent comm tracking
    broadcastCount:0,
    silentTicks:0,
    // role (emergent)
    role:"scout",         // scout|guardian|relay|coordinator
  };
}

function makeZone(id){
  const types=["PATROL","DENY","RECON"];
  const margin=100;
  return{
    id, type:types[id],
    x:rnd(margin,CFG.W-margin), y:rnd(margin,CFG.H-margin),
    r:rnd(55,90),
    priority:[0.6,1.0,0.8][id],
    cleared:false, contested:false,
    agentsInside:0,
  };
}

function makeThreat(id){
  return{
    id,
    x:rnd(60,CFG.W-60), y:rnd(60,CFG.H-60),
    vx:rnd(-0.8,0.8), vy:rnd(-0.8,0.8),
    type:["SURFACE","SUBSURFACE","FAST-BOAT","UNKNOWN"][id%4],
    detected:false, neutralised:false,
    detectedBy:null, detectionTick:null,
    threatLevel:rnd(0.4,1),
  };
}

function initState(){
  return{
    agents:   Array.from({length:CFG.N_AGENTS},(_,i)=>makeAgent(i)),
    zones:    Array.from({length:CFG.N_ZONES}, (_,i)=>makeZone(i)),
    threats:  Array.from({length:CFG.N_THREATS},(_,i)=>makeThreat(i)),
    tick:0,
    episode:0,
    phase:"DEPLOY",      // DEPLOY|SEARCH|COORDINATE|ENGAGE|DEBRIEF
    totalReward:0,
    events:[{id:0,tick:0,text:"REPLICANT-7 ONLINE — RL POLICY INITIALISED",kind:"sys"}],
    commLog:[],
    coverageGrid:new Float32Array(32*32),
    globalEpsilon:CFG.EPSILON_START,
    episodeRewards:[],
    collisionEvents:0,
    commEfficiency:0,
  };
}

// ─── SIMULATION TICK ─────────────────────────────────────────────────────────
function tick(state){
  const {agents,zones,threats,tick:t,coverageGrid} = state;
  let evId=t*500; const newEvents=[]; const newComms=[];

  // Clone
  const ag  = agents.map(a=>({...a,trail:[...a.trail],msgIn:[],msgOut:[],
    policy:{...a.policy,rewardHistory:[...a.policy.rewardHistory]}}));
  const zo  = zones.map(z=>({...z,agentsInside:0,contested:false}));
  const th  = threats.map(t=>({...t}));

  // ── 1. COMMUNICATION (bandwidth-limited) ─────────────────────────────────
  let totalComms=0, usefulComms=0;
  ag.forEach((a,i)=>{
    // Decide whether to broadcast (RL policy: emit only when info is novel)
    const shouldBroadcast = a.policy.epsilon < 0.5
      ? Math.random() < (1-a.policy.epsilon)*0.6   // learned: sparse
      : Math.random() < 0.9;                         // exploring: noisy

    if(!shouldBroadcast){a.silentTicks++;return;}

    let sent=0;
    ag.forEach((b,j)=>{
      if(i===j||sent>=CFG.COMM_BUDGET) return;
      if(d2(a,b)>CFG.COMM_RANGE) return;
      const attn=1-(d2(a,b)/CFG.COMM_RANGE)*0.5;
      ag[j].msgIn.push({from:i,detectedThreats:th.filter(t=>t.detectedBy===i).map(t=>t.id),
        pos:{x:a.x,y:a.y}, attn, tick:t});
      sent++; totalComms++;
      if(th.some(t=>t.detectedBy===i)) usefulComms++;
    });
    a.broadcastCount++;
  });

  // ── 2. THREAT DETECTION ──────────────────────────────────────────────────
  ag.forEach((a,i)=>{
    th.forEach((threat,ti)=>{
      const dd=d2(a,threat);
      if(dd<CFG.SENSOR_R&&!threat.neutralised){
        if(!threat.detected){
          th[ti]={...th[ti],detected:true,detectedBy:i,detectionTick:t};
          newEvents.push({id:evId++,tick:t,
            text:`A${i} DETECTED ${threat.type} T${ti} [${(threat.threatLevel*100).toFixed(0)}% THREAT]`,
            kind:"warn"});
          newComms.push({tick:t,from:i,msg:`THREAT ${ti} AT (${threat.x.toFixed(0)},${threat.y.toFixed(0)})`});
        }
        // neutralise if multiple agents converge
        const agentsNear=ag.filter(b=>d2(b,threat)<CFG.SENSOR_R*0.6).length;
        if(agentsNear>=2&&!threat.neutralised&&threat.detected){
          th[ti]={...th[ti],neutralised:true};
          newEvents.push({id:evId++,tick:t,
            text:`THREAT ${ti} NEUTRALISED — SWARM COORDINATION`,kind:"success"});
        }
      }
    });
  });

  // ── 3. ZONE UPDATE ───────────────────────────────────────────────────────
  zo.forEach((z,zi)=>{
    ag.forEach((a,ai)=>{
      if(d2(a,z)<z.r){
        zo[zi].agentsInside++;
        ag[ai].coverageScore+=z.priority*0.01;
      }
    });
    // Threat in zone = contested
    th.forEach(threat=>{
      if(!threat.neutralised&&d2(threat,z)<z.r) zo[zi].contested=true;
    });
    zo[zi].cleared = zo[zi].agentsInside>=2&&!zo[zi].contested;
  });

  // ── 4. RL REWARD COMPUTATION ─────────────────────────────────────────────
  const zoneCoverage  = zo.filter(z=>z.cleared).length/CFG.N_ZONES;
  const threatsFound  = th.filter(t=>t.detected).length/CFG.N_THREATS;
  const commPenalty   = (totalComms/Math.max(1,ag.length)) * Math.abs(CFG.REWARD_COMM_PENALTY);
  const baseReward    = zoneCoverage*CFG.REWARD_COVERAGE + threatsFound*1.5 - commPenalty;

  ag.forEach((a,i)=>{
    // Per-agent reward: local coverage + threat detection - excessive comms
    const localZone = zo.find(z=>d2(a,z)<z.r);
    const localR    = (localZone?localZone.priority*CFG.REWARD_COVERAGE:0)
                    + (th.some(t=>t.detectedBy===i)?1.2:0)
                    - (a.broadcastCount>CFG.COMM_BUDGET?0.3:0);
    ag[i].policy.update(localR);
  });

  // ── 5. MOVEMENT (policy-driven) ──────────────────────────────────────────
  ag.forEach((a,i)=>{
    if(a.battery<=0){ag[i].status="RTB";return;}
    const cur=noise(a.x,a.y,t);
    let fx=cur.x, fy=cur.y;

    // Determine objective from RL policy (greedy vs explore)
    const greedy = Math.random()>a.policy.epsilon;

    if(greedy){
      // Exploit: move toward assigned zone or detected threat
      const myZone=zo[a.zoneAssigned];
      const knownThreat=th.find(tr=>tr.detected&&!tr.neutralised);

      if(knownThreat&&a.status==="intercept"){
        const dx=knownThreat.x-a.x, dy=knownThreat.y-a.y, dd=Math.hypot(dx,dy)||1;
        fx+=dx/dd*2.0; fy+=dy/dd*2.0;
      } else {
        const tx=myZone.x+rnd(-20,20), ty=myZone.y+rnd(-20,20);
        const dx=tx-a.x, dy=ty-a.y, dd=Math.hypot(dx,dy)||1;
        fx+=dx/dd*1.6; fy+=dy/dd*1.6;
      }
    } else {
      // Explore: Lévy-flight random walk
      if(t%(25+i*8)===0){ag[i].vx=rnd(-1.8,1.8);ag[i].vy=rnd(-1.8,1.8);}
      fx+=rnd(-0.14,0.14); fy+=rnd(-0.14,0.14);
    }

    // Use incoming messages to adjust (learned comm utilisation)
    a.msgIn.forEach(msg=>{
      if(msg.detectedThreats.length>0&&Math.random()<0.7){
        const knownThr=th.find(tr=>msg.detectedThreats.includes(tr.id));
        if(knownThr&&!knownThr.neutralised){
          ag[i].status="intercept";
          const dx=knownThr.x-a.x,dy=knownThr.y-a.y,dd=Math.hypot(dx,dy)||1;
          fx+=dx/dd*0.8; fy+=dy/dd*0.8;
        }
      }
    });

    // Separation
    let collFlag=false;
    ag.forEach((b,j)=>{
      if(i===j)return;
      const dd=d2(a,b);
      if(dd<CFG.SEP_DIST&&dd>0){
        const f=(CFG.SEP_DIST-dd)/CFG.SEP_DIST*0.9;
        fx-=(b.x-a.x)/dd*f; fy-=(b.y-a.y)/dd*f;
        if(dd<20)collFlag=true;
      }
    });
    if(collFlag)ag[i].collisions++;

    // Speed cap
    let nvx=lerp(a.vx,a.vx+fx,.22), nvy=lerp(a.vy,a.vy+fy,.22);
    const ms=a.status==="intercept"?2.5:1.8;
    const s=Math.hypot(nvx,nvy);
    if(s>ms){nvx=nvx/s*ms;nvy=nvy/s*ms;}
    if(s<0.15&&a.status!=="RTB"){nvx=rnd(-0.8,0.8);nvy=rnd(-0.8,0.8);}

    // Wall
    let nx=a.x+nvx,ny=a.y+nvy;
    const m=22;
    if(nx<m){nvx=Math.abs(nvx)*.7;nx=m;}
    if(nx>CFG.W-m){nvx=-Math.abs(nvx)*.7;nx=CFG.W-m;}
    if(ny<m){nvy=Math.abs(nvy)*.7;ny=m;}
    if(ny>CFG.H-m){nvy=-Math.abs(nvy)*.7;ny=CFG.H-m;}

    // Trail
    const trail=[{x:a.x,y:a.y},...a.trail].slice(0,CFG.TRAIL);

    // Battery
    const nb=Math.max(0,a.battery-0.005-Math.hypot(nvx,nvy)*0.0006);
    if(nb<12&&a.status!=="RTB"){
      ag[i].status="RTB";
      newEvents.push({id:evId++,tick:t,text:`A${i} LOW ENERGY — RETURNING TO BASE`,kind:"warn"});
    }

    // Emergent role assignment based on behaviour pattern
    let role=a.role;
    if(a.broadcastCount>30&&!greedy) role="relay";
    else if(a.status==="intercept") role="guardian";
    else if(a.policy.exploitRate>0.7) role="coordinator";
    else role="scout";

    ag[i]={...ag[i],x:nx,y:ny,vx:nvx,vy:nvy,trail,battery:nb,role,
      status:ag[i].status!=="RTB"?ag[i].status:a.status};
  });

  // ── 6. THREAT MOVEMENT ────────────────────────────────────────────────────
  th.forEach((threat,i)=>{
    if(threat.neutralised)return;
    const cur=noise(threat.x,threat.y,t*1.3);
    let nx=threat.x+threat.vx+cur.x*0.3;
    let ny=threat.y+threat.vy+cur.y*0.3;
    if(nx<30||nx>CFG.W-30)th[i].vx*=-1;
    if(ny<30||ny>CFG.H-30)th[i].vy*=-1;
    th[i]={...th[i],x:clamp(nx,30,CFG.W-30),y:clamp(ny,30,CFG.H-30)};
  });

  // ── 7. COVERAGE GRID ─────────────────────────────────────────────────────
  const ng=new Float32Array(coverageGrid);
  ag.forEach(a=>{
    const gx=Math.floor((a.x/CFG.W)*32), gy=Math.floor((a.y/CFG.H)*32);
    const idx=clamp(gy,0,31)*32+clamp(gx,0,31);
    ng[idx]=Math.min(1,(ng[idx]||0)+0.03);
  });
  for(let i=0;i<ng.length;i++)ng[i]*=0.997;

  // ── 8. PHASE & GLOBAL STATS ───────────────────────────────────────────────
  const anyDetected=th.some(t=>t.detected);
  const allNeutralised=th.every(t=>t.neutralised);
  let phase=state.phase;
  if(t<30)phase="DEPLOY";
  else if(!anyDetected)phase="SEARCH";
  else if(anyDetected&&!allNeutralised)phase="COORDINATE";
  else if(allNeutralised)phase="DEBRIEF";

  const globalEps=ag.reduce((s,a)=>s+a.policy.epsilon,0)/ag.length;
  const commEff=totalComms>0?usefulComms/totalComms:0;

  // Episode boundary every 300 ticks
  let episode=state.episode;
  let episodeRewards=[...state.episodeRewards];
  if(t>0&&t%300===0){
    episode++;
    const epR=ag.reduce((s,a)=>s+a.policy.cumReward,0)/ag.length;
    episodeRewards=[...episodeRewards,{ep:episode,r:epR}].slice(-20);
    newEvents.push({id:evId++,tick:t,
      text:`EPISODE ${episode} — AVG REWARD ${epR.toFixed(2)} — ε=${globalEps.toFixed(3)}`,
      kind:"info"});
  }

  return{
    ...state,agents:ag,zones:zo,threats:th,tick:t+1,phase,
    globalEpsilon:globalEps,episode,episodeRewards,
    totalReward:state.totalReward+baseReward,
    events:[...newEvents,...state.events].slice(0,80),
    commLog:[...newComms,...state.commLog].slice(0,30),
    coverageGrid:ng,
    commEfficiency:lerp(state.commEfficiency,commEff,.08),
    collisionEvents:state.collisionEvents+ag.filter(a=>a.collisions>0).length,
  };
}

// ─── SVG COMPONENTS ──────────────────────────────────────────────────────────
function ZoneMarker({zone}){
  const col=zone.cleared?D.g:zone.contested?D.r:D.b;
  const typeIcon={"PATROL":"◈","DENY":"⬡","RECON":"◎"}[zone.type]||"●";
  return <g transform={`translate(${zone.x},${zone.y})`}>
    <circle r={zone.r} fill={`${col}09`} stroke={col}
      strokeWidth={zone.contested?1.8:.8}
      strokeDasharray={zone.cleared?"":"5 4"} opacity=".7"/>
    <text textAnchor="middle" dy=".3em" fontSize="9" fill={col}
      fontFamily={D.FD} letterSpacing="2" opacity=".85">
      {typeIcon} {zone.type}
    </text>
    {zone.contested&&<circle r={zone.r+8} fill="none" stroke={D.r}
      strokeWidth=".4" opacity=".3" strokeDasharray="2 6"/>}
    <text y={zone.r+14} textAnchor="middle" fontSize="7" fill={col}
      fontFamily={D.FM} opacity=".6">
      {zone.agentsInside} UNITS
    </text>
  </g>;
}

function ThreatMarker({threat}){
  if(threat.neutralised)return(
    <g transform={`translate(${threat.x},${threat.y})`} opacity=".3">
      <circle r={8} fill="none" stroke={D.g} strokeWidth="1" strokeDasharray="2 3"/>
      <line x1={-5} y1={-5} x2={5} y2={5} stroke={D.g} strokeWidth="1.5"/>
      <line x1={5} y1={-5} x2={-5} y2={5} stroke={D.g} strokeWidth="1.5"/>
    </g>
  );
  const col=threat.detected?D.r:D.y;
  return <g transform={`translate(${threat.x},${threat.y})`}>
    {threat.detected&&<>
      <circle r={16} fill="none" stroke={D.r} strokeWidth=".4" opacity=".3"/>
      <circle r={22} fill="none" stroke={D.r} strokeWidth=".2" opacity=".15"/>
    </>}
    <polygon points="0,-10 9,6 -9,6" fill={`${col}28`} stroke={col}
      strokeWidth={threat.detected?1.8:1}/>
    <text y={16} textAnchor="middle" fontSize="6.5" fill={col}
      fontFamily={D.FM}>{threat.type}</text>
    <text y={-14} textAnchor="middle" fontSize="7" fill={col}
      fontFamily={D.FD}>{`T${threat.id}`}</text>
  </g>;
}

function AgentNode({agent,selected,onClick}){
  const col=agent.col;
  const active=agent.battery>5&&agent.status!=="RTB";
  return <g transform={`translate(${agent.x},${agent.y})`}
    style={{cursor:"pointer"}} onClick={onClick}>
    {selected&&<circle r={CFG.COMM_RANGE} fill="none" stroke={col}
      strokeWidth=".4" strokeDasharray="4 7" opacity=".22"/>}
    {selected&&<circle r={CFG.SENSOR_R} fill={`${col}0a`} stroke={col}
      strokeWidth=".7" opacity=".5"/>}
    {/* Policy confidence ring */}
    <circle r={11} fill="none" stroke={col} strokeWidth="1.5" opacity=".5"
      strokeDasharray={`${agent.policy.exploitRate*69} 69`}
      transform="rotate(-90)"/>
    <circle r={8} fill={`${col}1a`} stroke={col}
      strokeWidth={selected?2:1} opacity={active?1:.35}/>
    <circle r={3.5} fill={col} opacity={active?1:.25}/>
    {active&&<line x1={0} y1={0} x2={agent.vx*10} y2={agent.vy*10}
      stroke={col} strokeWidth="1.4" opacity=".7"/>}
    {/* Role badge */}
    <text y={-13} textAnchor="middle" fontSize="7" fill={col}
      fontFamily={D.FD} letterSpacing="1">{`A${agent.id}`}</text>
    <text y={16} textAnchor="middle" fontSize="5.5" fill={col}
      fontFamily={D.FM} opacity=".7">{agent.role.toUpperCase()}</text>
    {/* Comm activity flash */}
    {agent.msgOut&&agent.msgOut.length>0&&
      <circle r={13} fill="none" stroke={col} strokeWidth=".6" opacity=".5"/>}
    {/* Battery strip */}
    <g transform="translate(-5,18)">
      <rect width={10} height={2} rx="1" fill={D.muted}/>
      <rect width={agent.battery/100*10} height={2} rx="1"
        fill={agent.battery>40?D.g:agent.battery>20?D.y:D.r}/>
    </g>
  </g>;
}

function CommLinks({agents}){
  const lines=[];
  agents.forEach((a,i)=>agents.slice(i+1).forEach((b,j)=>{
    const dd=d2(a,b); if(dd>CFG.COMM_RANGE)return;
    const str=1-dd/CFG.COMM_RANGE;
    const active=a.msgIn.some(m=>m.from===i+1+j)||b.msgIn.some(m=>m.from===i);
    lines.push(<line key={`${i}-${j}`}
      x1={a.x} y1={a.y} x2={b.x} y2={b.y}
      stroke={active?D.b:D.muted}
      strokeWidth={active?str*.9:.25}
      opacity={active?str*.5:str*.08}
      strokeDasharray={active?undefined:"2 9"}/>);
  }));
  return <g>{lines}</g>;
}

function AgentTrails({agents}){
  return <g>{agents.map(a=>{
    if(a.trail.length<2)return null;
    const pts=[{x:a.x,y:a.y},...a.trail];
    return <g key={a.id}>{pts.slice(0,-1).map((p,i)=>{
      const q=pts[i+1];
      return <line key={i} x1={p.x} y1={p.y} x2={q.x} y2={q.y}
        stroke={a.col} strokeWidth={(1-i/pts.length)*1.4}
        opacity={(1-i/pts.length)*.3}/>;
    })}</g>;
  })}</g>;
}

function CoverageGrid({grid}){
  const cs=CFG.W/32;
  return <g opacity=".18">{Array.from(grid).map((v,i)=>{
    if(v<.05)return null;
    return <rect key={i} x={(i%32)*cs} y={Math.floor(i/32)*cs}
      width={cs} height={cs} fill={D.g} opacity={v*.8}/>;
  })}</g>;
}

// ─── UI ATOMS ────────────────────────────────────────────────────────────────
const SC = s=>({
  standby:D.textDim,patrol:D.b,intercept:D.r,
  cover:D.g,RTB:D.o,idle:D.textDim,
}[s]||D.text);

const RC = r=>({scout:D.b,guardian:D.r,relay:D.y,coordinator:D.g}[r]||D.text);

function Panel({title,children,style={}}){
  return <div style={{background:D.bg1,border:`1px solid ${D.border}`,
    borderRadius:4,overflow:"hidden",display:"flex",flexDirection:"column",
    minHeight:0,...style}}>
    {title&&<div style={{padding:"4px 10px",fontSize:7.5,letterSpacing:3,
      color:D.textDim,borderBottom:`1px solid ${D.border}`,
      background:D.bg0,fontFamily:D.FD,flexShrink:0}}>{title}</div>}
    <div style={{flex:1,overflow:"auto",minHeight:0}}>{children}</div>
  </div>;
}

function Metric({label,value,sub,col=D.b,small}){
  return <div style={{background:D.bg0,border:`1px solid ${D.border}`,
    borderRadius:4,padding:small?"5px 8px":"7px 10px"}}>
    <div style={{fontSize:6.5,color:D.textDim,letterSpacing:3,
      marginBottom:2,fontFamily:D.FD}}>{label}</div>
    <div style={{fontSize:small?13:16,color:col,fontFamily:D.FD,
      fontWeight:700,lineHeight:1}}>{value}</div>
    {sub&&<div style={{fontSize:7,color:D.textDim,marginTop:1,
      fontFamily:D.FM}}>{sub}</div>}
  </div>;
}

function Btn({onClick,children,active,col=D.b,small}){
  return <button onClick={onClick} style={{
    background:active?`${col}22`:D.bg0,
    border:`1px solid ${active?col:D.border}`,
    color:active?col:D.textDim,
    padding:small?"3px 8px":"5px 13px",
    fontSize:small?7:8,letterSpacing:2,cursor:"pointer",borderRadius:3,
    fontFamily:D.FD,transition:"all .14s",
    display:"flex",alignItems:"center",gap:4,
  }}>{children}</button>;
}

function MiniRewardChart({history,col}){
  if(!history||history.length<2)return(
    <div style={{height:36,display:"flex",alignItems:"center",
      justifyContent:"center",fontSize:7,color:D.textDim}}>
      ACCUMULATING DATA…
    </div>
  );
  const mn=Math.min(...history), mx=Math.max(...history);
  const range=mx-mn||1;
  const W2=120,H2=36;
  const pts=history.map((v,i)=>
    `${(i/(history.length-1))*W2},${H2-((v-mn)/range)*(H2-4)-2}`
  ).join(" ");
  return <svg width={W2} height={H2}>
    <polyline points={pts} fill="none" stroke={col||D.g} strokeWidth="1.2" opacity=".8"/>
    <line x1={0} y1={H2-2} x2={W2} y2={H2-2} stroke={D.border} strokeWidth=".5"/>
  </svg>;
}

function EpisodeChart({episodeRewards}){
  if(!episodeRewards||episodeRewards.length<2)return(
    <div style={{padding:8,fontSize:7.5,color:D.textDim,fontFamily:D.FM}}>
      AWAITING EPISODE DATA…
    </div>
  );
  const vals=episodeRewards.map(e=>e.r);
  const mn=Math.min(...vals), mx=Math.max(...vals);
  const range=mx-mn||1;
  const W2=200,H2=60;
  const pts=vals.map((v,i)=>
    `${(i/(vals.length-1))*W2},${H2-((v-mn)/range)*(H2-6)-3}`
  ).join(" ");
  return <svg width={W2} height={H2} style={{display:"block"}}>
    <defs>
      <linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor={D.g} stopOpacity=".3"/>
        <stop offset="100%" stopColor={D.g} stopOpacity="0"/>
      </linearGradient>
    </defs>
    <rect width={W2} height={H2} fill={D.bg0} rx="2"/>
    <polyline points={pts+" "+`${W2},${H2} 0,${H2}`}
      fill="url(#rg)" stroke="none"/>
    <polyline points={pts} fill="none" stroke={D.g} strokeWidth="1.5"/>
    <text x={2} y={10} fontSize="6" fill={D.textDim} fontFamily={D.FM}>
      {mx.toFixed(1)}
    </text>
    <text x={2} y={H2-2} fontSize="6" fill={D.textDim} fontFamily={D.FM}>
      {mn.toFixed(1)}
    </text>
  </svg>;
}

function PhaseBar({phase}){
  const phases=["DEPLOY","SEARCH","COORDINATE","ENGAGE","DEBRIEF"];
  const cols=[D.textDim,D.b,D.y,D.r,D.g];
  const idx=phases.indexOf(phase);
  return <div style={{display:"flex",gap:3,alignItems:"center"}}>
    {phases.map((p,i)=><div key={p} style={{height:3.5,
      width:i===idx?26:12,borderRadius:2,
      background:i<=idx?cols[i]:D.muted,
      transition:"width .35s,background .35s"}}/>)}
    <span style={{fontSize:8,color:cols[idx]||D.textDim,fontFamily:D.FD,
      letterSpacing:2,marginLeft:5}}>{phase}</span>
  </div>;
}

function EpsilonGauge({epsilon}){
  const pct=epsilon*100;
  const col=epsilon>.6?D.r:epsilon>.3?D.y:D.g;
  return <div style={{display:"flex",alignItems:"center",gap:6}}>
    <span style={{fontSize:7,color:D.textDim,letterSpacing:2,
      fontFamily:D.FD}}>ε-GREEDY</span>
    <div style={{width:64,height:5,background:D.muted,borderRadius:3,overflow:"hidden"}}>
      <div style={{width:`${pct}%`,height:"100%",background:col,transition:"width .3s"}}/>
    </div>
    <span style={{fontSize:8,color:col,fontFamily:D.FM}}>{epsilon.toFixed(3)}</span>
  </div>;
}

function AgentDetailPanel({agent,agents}){
  if(!agent)return(
    <div style={{padding:14,color:D.textDim,fontSize:8.5,fontFamily:D.FM,
      textAlign:"center",paddingTop:28}}>
      SELECT AN AGENT NODE TO INSPECT
    </div>
  );
  const links=agents.filter(b=>b.id!==agent.id&&d2(agent,b)<CFG.COMM_RANGE).length;
  return <div style={{padding:"9px 10px",fontSize:8.5,fontFamily:D.FM}}>
    <div style={{color:agent.col,fontSize:13,fontFamily:D.FD,
      letterSpacing:3,marginBottom:8}}>
      AGENT-{agent.id}
      <span style={{marginLeft:8,fontSize:8,color:RC(agent.role),
        background:`${RC(agent.role)}18`,padding:"2px 6px",
        borderRadius:10,border:`1px solid ${RC(agent.role)}55`}}>
        {agent.role.toUpperCase()}
      </span>
    </div>
    {[
      ["STATUS",       agent.status.toUpperCase(),     SC(agent.status)],
      ["POLICY ε",     agent.policy.epsilon.toFixed(4), agent.policy.epsilon>.5?D.r:D.g],
      ["EXPLOIT RATE", `${(agent.policy.exploitRate*100).toFixed(1)}%`,D.g],
      ["CUM. REWARD",  agent.policy.cumReward.toFixed(2),D.y],
      ["ACTOR LOSS",   agent.policy.actorLoss.toFixed(4),D.o],
      ["CRITIC LOSS",  agent.policy.criticLoss.toFixed(4),D.p],
      ["MESH LINKS",   links,links>2?D.g:D.textDim],
      ["BROADCASTS",   agent.broadcastCount],
      ["SILENT TICKS", agent.silentTicks,D.textDim],
      ["BATTERY",      `${agent.battery.toFixed(0)}%`,
        agent.battery>40?D.g:agent.battery>20?D.y:D.r],
    ].map(([k,v,c])=>(
      <div key={k} style={{display:"flex",justifyContent:"space-between",
        padding:"2.5px 0",borderBottom:`1px solid ${D.border}22`}}>
        <span style={{color:D.textDim}}>{k}</span>
        <span style={{color:c||D.text,fontWeight:"bold"}}>{v}</span>
      </div>
    ))}
    <div style={{marginTop:8,fontSize:7,color:D.textDim,letterSpacing:2,
      fontFamily:D.FD,marginBottom:4}}>REWARD CURVE</div>
    <MiniRewardChart history={agent.policy.rewardHistory} col={agent.col}/>
  </div>;
}

function CommLogPanel({log}){
  if(!log||log.length===0)return(
    <div style={{padding:8,fontSize:8,color:D.textDim,fontFamily:D.FM}}>
      COMM CHANNEL SILENT
    </div>
  );
  return <div style={{padding:"4px 8px"}}>
    {log.map((c,i)=>(
      <div key={i} style={{fontSize:8,color:i===0?D.b:D.textDim,
        padding:"2px 0",borderBottom:`1px solid ${D.border}18`,
        fontFamily:D.FM}}>
        <span style={{color:D.textDim,marginRight:8}}>[{c.tick}]</span>
        <span style={{color:D.b,marginRight:6}}>A{c.from}</span>
        {c.msg}
      </div>
    ))}
  </div>;
}

function PolicyTable({agents}){
  return <table style={{width:"100%",borderCollapse:"collapse",fontSize:8}}>
    <thead><tr style={{background:D.bg0}}>
      {["ID","ROLE","ε","EXPLOIT%","CUM-R","ACT-L","CRIT-L","BCAST"].map(h=>(
        <th key={h} style={{padding:"3px 7px",textAlign:"left",
          color:D.textDim,fontSize:6.5,letterSpacing:2,
          borderBottom:`1px solid ${D.border}`,fontFamily:D.FD,fontWeight:"normal"}}>
          {h}
        </th>
      ))}
    </tr></thead>
    <tbody>{agents.map(a=>(
      <tr key={a.id} style={{borderBottom:`1px solid ${D.border}1a`}}>
        <td style={{padding:"3px 7px",color:a.col,fontFamily:D.FD}}>A{a.id}</td>
        <td style={{padding:"3px 7px"}}>
          <span style={{color:RC(a.role),fontSize:7,
            background:`${RC(a.role)}15`,padding:"1px 5px",borderRadius:8}}>
            {a.role}
          </span>
        </td>
        <td style={{padding:"3px 7px",color:a.policy.epsilon>.5?D.r:D.g,fontFamily:D.FM}}>
          {a.policy.epsilon.toFixed(3)}
        </td>
        <td style={{padding:"3px 7px",fontFamily:D.FM}}>
          <div style={{display:"flex",alignItems:"center",gap:3}}>
            <div style={{width:36,height:4,background:D.muted,borderRadius:2,overflow:"hidden"}}>
              <div style={{width:`${a.policy.exploitRate*100}%`,height:"100%",background:D.g}}/>
            </div>
            <span style={{color:D.textDim,fontSize:7}}>{(a.policy.exploitRate*100).toFixed(0)}%</span>
          </div>
        </td>
        <td style={{padding:"3px 7px",color:D.y,fontFamily:D.FM}}>
          {a.policy.cumReward.toFixed(1)}
        </td>
        <td style={{padding:"3px 7px",color:D.o,fontFamily:D.FM}}>
          {a.policy.actorLoss.toFixed(3)}
        </td>
        <td style={{padding:"3px 7px",color:D.p,fontFamily:D.FM}}>
          {a.policy.criticLoss.toFixed(3)}
        </td>
        <td style={{padding:"3px 7px",color:D.textDim,fontFamily:D.FM}}>
          {a.broadcastCount}
        </td>
      </tr>
    ))}</tbody>
  </table>;
}

function EmergentBehaviourPanel({agents,state}){
  const roles=["scout","guardian","relay","coordinator"];
  const roleCounts=roles.map(r=>({
    role:r,col:RC(r),
    count:agents.filter(a=>a.role===r).length,
  }));
  const avgEps=agents.reduce((s,a)=>s+a.policy.epsilon,0)/agents.length;
  const avgExploit=agents.reduce((s,a)=>s+a.policy.exploitRate,0)/agents.length;
  return <div style={{padding:"8px 10px",fontSize:8.5,fontFamily:D.FM}}>
    <div style={{fontSize:7,color:D.textDim,letterSpacing:3,
      fontFamily:D.FD,marginBottom:8}}>EMERGENT ROLE DISTRIBUTION</div>
    {roleCounts.map(({role,col,count})=>(
      <div key={role} style={{display:"flex",alignItems:"center",gap:6,marginBottom:5}}>
        <div style={{width:8,height:8,borderRadius:"50%",background:col,flexShrink:0}}/>
        <span style={{color:col,minWidth:90,fontSize:8,fontFamily:D.FM}}>{role.toUpperCase()}</span>
        <div style={{flex:1,height:5,background:D.muted,borderRadius:2,overflow:"hidden"}}>
          <div style={{width:`${(count/agents.length)*100}%`,height:"100%",background:col}}/>
        </div>
        <span style={{color:D.textDim,minWidth:16,textAlign:"right"}}>{count}</span>
      </div>
    ))}
    <div style={{marginTop:10,borderTop:`1px solid ${D.border}`,paddingTop:8}}>
      <div style={{fontSize:7,color:D.textDim,letterSpacing:3,
        fontFamily:D.FD,marginBottom:6}}>COMM BANDWIDTH USAGE</div>
      <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:4}}>
        <span style={{color:D.textDim,minWidth:90}}>EFFICIENCY</span>
        <div style={{flex:1,height:5,background:D.muted,borderRadius:2,overflow:"hidden"}}>
          <div style={{width:`${state.commEfficiency*100}%`,height:"100%",background:D.g}}/>
        </div>
        <span style={{color:D.g,minWidth:32,textAlign:"right"}}>
          {(state.commEfficiency*100).toFixed(0)}%
        </span>
      </div>
      <div style={{display:"flex",alignItems:"center",gap:6}}>
        <span style={{color:D.textDim,minWidth:90}}>EXPLORATION</span>
        <div style={{flex:1,height:5,background:D.muted,borderRadius:2,overflow:"hidden"}}>
          <div style={{width:`${(1-avgExploit)*100}%`,height:"100%",background:D.r}}/>
        </div>
        <span style={{color:D.r,minWidth:32,textAlign:"right"}}>
          {((1-avgExploit)*100).toFixed(0)}%
        </span>
      </div>
    </div>
    <div style={{marginTop:10,borderTop:`1px solid ${D.border}`,paddingTop:8}}>
      <div style={{fontSize:7,color:D.textDim,letterSpacing:3,
        fontFamily:D.FD,marginBottom:6}}>EPISODE REWARD CURVE</div>
      <EpisodeChart episodeRewards={state.episodeRewards}/>
      <div style={{fontSize:7,color:D.textDim,marginTop:3,fontFamily:D.FM}}>
        EPISODE {state.episode} — TOTAL Σ {state.totalReward.toFixed(1)}
      </div>
    </div>
  </div>;
}

function NodeBadge({agent,selected,onClick}){
  const col=agent.col;
  return <div onClick={onClick} style={{cursor:"pointer",
    background:selected?`${col}18`:D.bg2,
    border:`1px solid ${selected?col:D.border}`,
    borderRadius:3,padding:"3px 6px",
    display:"flex",flexDirection:"column",alignItems:"center",
    gap:2,minWidth:30,transition:"all .14s"}}>
    <div style={{display:"flex",alignItems:"center",gap:3}}>
      <div style={{width:5,height:5,borderRadius:"50%",background:col}}/>
      <span style={{fontSize:6.5,color:col,fontFamily:D.FD}}>A{agent.id}</span>
    </div>
    <div style={{width:24,height:2,background:D.muted,borderRadius:1,overflow:"hidden"}}>
      <div style={{width:`${agent.battery}%`,height:"100%",
        background:agent.battery>40?D.g:D.r}}/>
    </div>
    <div style={{fontSize:5.5,color:RC(agent.role),fontFamily:D.FM}}>
      {agent.role[0].toUpperCase()}
    </div>
  </div>;
}

// ─── ROOT APPLICATION ─────────────────────────────────────────────────────────
const GOOGLE_FONTS = `@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600&display=swap');`;

const STYLES = `
${GOOGLE_FONTS}
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b0e14;overflow:hidden}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:#0b0e14}
::-webkit-scrollbar-thumb{background:#1a2d42;border-radius:2px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.1}}
@keyframes slide-in{from{transform:translateX(12px);opacity:0}to{transform:translateX(0);opacity:1}}
.blink{animation:blink 1.4s ease infinite}
`;

export default function ReplicantSwarm(){
  const [sim,setSim]     = useState(initState);
  const [running,setRun] = useState(false);
  const [sel,setSel]     = useState(null);
  const [speed,setSpeed] = useState(1);
  const [tab,setTab]     = useState("INTEL");
  const [layers,setLayers] = useState({
    zones:true,threats:true,links:true,trails:true,coverage:false,
  });
  const ivRef=useRef(null);
  const toggleLayer=useCallback(k=>setLayers(l=>({...l,[k]:!l[k]})),[]);
  const reset=useCallback(()=>{setRun(false);setSim(initState());setSel(null);},[]);

  useEffect(()=>{
    if(!running){clearInterval(ivRef.current);return;}
    clearInterval(ivRef.current);
    ivRef.current=setInterval(()=>{
      setSim(prev=>{let s=prev;for(let i=0;i<speed;i++)s=tick(s);return s;});
    },CFG.TICK_MS);
    return()=>clearInterval(ivRef.current);
  },[running,speed]);

  const {agents,zones,threats,coverageGrid,events,commLog}=sim;
  const selAgent=agents.find(a=>a.id===sel);
  const activeN=agents.filter(a=>a.battery>5).length;
  const meshLinks=agents.reduce((c,a,i)=>c+agents.slice(i+1)
    .filter(b=>d2(a,b)<CFG.COMM_RANGE).length,0);
  const threatsNeutralised=threats.filter(t=>t.neutralised).length;
  const TABS=["INTEL","POLICY","COMMS","EMERGENT"];
  const PHASE_C=p=>({DEPLOY:D.textDim,SEARCH:D.b,COORDINATE:D.y,
    ENGAGE:D.r,DEBRIEF:D.g}[p]||D.text);

  return <>
    <style>{STYLES}</style>
    <div style={{width:"100vw",height:"100vh",background:D.bg0,
      display:"flex",flexDirection:"column",color:D.text,
      fontFamily:D.FM,overflow:"hidden"}}>

      {/* ══ HEADER ══════════════════════════════════════════════════════ */}
      <div style={{height:50,background:D.bg0,
        borderBottom:`1px solid ${D.border}`,
        display:"flex",alignItems:"center",padding:"0 16px",
        gap:16,flexShrink:0}}>
        <div>
          <div style={{fontFamily:D.FD,fontSize:14,fontWeight:900,
            color:D.b,letterSpacing:5,lineHeight:1}}>
            ◈ REPLICANT-7
          </div>
          <div style={{fontSize:6.5,color:D.textDim,letterSpacing:4}}>
            MULTI-AGENT RL SWARM — MADDPG · SPARSE COMMS · SARONIC-PARADIGM
          </div>
        </div>
        <PhaseBar phase={sim.phase}/>
        <div style={{flex:1}}/>

        {/* Quick stats */}
        {[
          ["TICK",    sim.tick,         D.b],
          ["EPISODE", sim.episode,      D.y],
          ["ACTIVE",  `${activeN}/${CFG.N_AGENTS}`, activeN>5?D.g:D.o],
          ["THREATS", `${threatsNeutralised}/${CFG.N_THREATS}`, D.r],
          ["LINKS",   `${meshLinks}L`,  meshLinks>6?D.g:D.o],
        ].map(([l,v,c])=>(
          <div key={l} style={{background:D.bg2,border:`1px solid ${D.border}`,
            borderRadius:3,padding:"2px 8px",display:"flex",gap:5,alignItems:"center"}}>
            <span style={{fontSize:6.5,color:D.textDim,letterSpacing:1.5,
              fontFamily:D.FD}}>{l}</span>
            <span style={{fontSize:9.5,color:c,fontFamily:D.FD,fontWeight:700}}>{v}</span>
          </div>
        ))}

        <EpsilonGauge epsilon={sim.globalEpsilon}/>

        <div style={{display:"flex",gap:5,marginLeft:8}}>
          <Btn onClick={()=>setRun(r=>!r)} active={running}
            col={running?D.o:D.g}>
            {running?"◼ HALT":"▶ TRAIN"}
          </Btn>
          <Btn onClick={reset} col={D.textDim}>⟳ RESET</Btn>
          {[1,3,6,12].map(s=><Btn key={s} small onClick={()=>setSpeed(s)}
            active={speed===s} col={D.b}>{`×${s}`}</Btn>)}
        </div>
      </div>

      {/* ══ BODY ════════════════════════════════════════════════════════ */}
      <div style={{flex:1,display:"flex",overflow:"hidden",minHeight:0}}>

        {/* ── MAP ───────────────────────────────────────────────────── */}
        <div style={{flex:"0 0 auto",display:"flex",flexDirection:"column",
          borderRight:`1px solid ${D.border}`}}>
          {/* Layer bar */}
          <div style={{height:28,background:D.bg0,
            borderBottom:`1px solid ${D.border}`,
            display:"flex",alignItems:"center",
            padding:"0 8px",gap:3,flexShrink:0}}>
            <span style={{fontSize:6.5,color:D.textDim,letterSpacing:2,
              marginRight:4,fontFamily:D.FD}}>LAYERS:</span>
            {Object.keys(layers).map(k=>(
              <button key={k} onClick={()=>toggleLayer(k)} style={{
                fontSize:6.5,padding:"2px 6px",cursor:"pointer",
                background:layers[k]?`${D.b}18`:"transparent",
                border:`1px solid ${layers[k]?D.b:D.border}`,
                color:layers[k]?D.b:D.textDim,borderRadius:2,
                letterSpacing:1.5,fontFamily:D.FM,
              }}>{k.toUpperCase()}</button>
            ))}
          </div>

          {/* SVG */}
          <svg width={CFG.W} height={CFG.H}
            style={{display:"block",background:D.bg0,flex:"0 0 auto"}}>
            <defs>
              <radialGradient id="mapbg" cx="50%" cy="50%" r="65%">
                <stop offset="0%" stopColor="#111a25"/>
                <stop offset="100%" stopColor="#0b0e14"/>
              </radialGradient>
              <radialGradient id="vignette" cx="50%" cy="50%" r="70%">
                <stop offset="55%" stopColor="transparent"/>
                <stop offset="100%" stopColor="#0b0e14" stopOpacity=".6"/>
              </radialGradient>
            </defs>
            <rect width={CFG.W} height={CFG.H} fill="url(#mapbg)"/>
            {/* Grid */}
            {Array.from({length:14},(_,i)=><g key={i}>
              <line x1={i*(CFG.W/13)} y1={0} x2={i*(CFG.W/13)} y2={CFG.H}
                stroke={D.grid} strokeWidth=".5"/>
              <line x1={0} y1={i*(CFG.H/13)} x2={CFG.W} y2={i*(CFG.H/13)}
                stroke={D.grid} strokeWidth=".5"/>
            </g>)}
            <rect x={1} y={1} width={CFG.W-2} height={CFG.H-2}
              fill="none" stroke={D.border} strokeWidth="1.5"/>
            {/* Corners */}
            {[[0,0],[CFG.W,0],[0,CFG.H],[CFG.W,CFG.H]].map(([cx,cy],i)=>(
              <g key={i} transform={`translate(${cx},${cy}) rotate(${i*90})`}>
                <line x1={0} y1={0} x2={20} y2={0} stroke={D.borderHi} strokeWidth="1.8"/>
                <line x1={0} y1={0} x2={0} y2={20} stroke={D.borderHi} strokeWidth="1.8"/>
              </g>
            ))}

            {/* Layers */}
            {layers.coverage&&<CoverageGrid grid={coverageGrid}/>}
            {layers.zones&&zones.map(z=><ZoneMarker key={z.id} zone={z}/>)}
            {layers.threats&&threats.map(t=><ThreatMarker key={t.id} threat={t}/>)}
            {layers.links&&<CommLinks agents={agents}/>}
            {layers.trails&&<AgentTrails agents={agents}/>}
            {agents.map(a=><AgentNode key={a.id} agent={a}
              selected={a.id===sel}
              onClick={()=>setSel(a.id===sel?null:a.id)}/>)}
            <rect width={CFG.W} height={CFG.H} fill="url(#vignette)"/>

            {/* Coord labels */}
            {[0,200,400,600].map(v=><g key={v}>
              <text x={v+3} y={11} fontSize="6.5" fill={D.textXd} fontFamily={D.FM}>{v}</text>
              <text x={3} y={v+10} fontSize="6.5" fill={D.textXd} fontFamily={D.FM}>{v}</text>
            </g>)}
          </svg>
        </div>

        {/* ── RIGHT PANEL ─────────────────────────────────────────── */}
        <div style={{flex:1,display:"flex",flexDirection:"column",
          overflow:"hidden",minWidth:0,gap:4,padding:4}}>

          {/* Metric row */}
          <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",
            gap:4,flexShrink:0}}>
            <Metric label="PHASE"       value={sim.phase}
              col={PHASE_C(sim.phase)}/>
            <Metric label="TOTAL Σ-R"   value={sim.totalReward.toFixed(1)}
              sub="cumulative" col={D.g}/>
            <Metric label="COMM EFF."   value={`${(sim.commEfficiency*100).toFixed(0)}%`}
              sub="useful/total" col={D.b}/>
            <Metric label="NEUTRALISED" value={`${threatsNeutralised}/${CFG.N_THREATS}`}
              col={threatsNeutralised===CFG.N_THREATS?D.g:D.r}/>
            <Metric label="ZONES CLR"
              value={`${zones.filter(z=>z.cleared).length}/${CFG.N_ZONES}`}
              col={D.y}/>
            <Metric label="COLLISIONS"  value={sim.collisionEvents}
              sub="total events" col={D.o}/>
          </div>

          {/* Tabs */}
          <div style={{flex:1,display:"flex",flexDirection:"column",
            overflow:"hidden",minHeight:0,border:`1px solid ${D.border}`,
            borderRadius:4,background:D.bg1}}>
            <div style={{display:"flex",flexShrink:0,
              borderBottom:`1px solid ${D.border}`,background:D.bg0}}>
              {TABS.map(tb=>(
                <button key={tb} onClick={()=>setTab(tb)} style={{
                  padding:"5px 14px",fontSize:7.5,letterSpacing:2,
                  cursor:"pointer",background:tab===tb?`${D.b}14`:"transparent",
                  border:"none",
                  borderBottom:tab===tb?`2px solid ${D.b}`:"2px solid transparent",
                  color:tab===tb?D.b:D.textDim,fontFamily:D.FD,
                }}>{tb}</button>
              ))}
            </div>
            <div style={{flex:1,overflow:"auto",minHeight:0}}>
              {tab==="INTEL"&&(
                <div style={{display:"flex",height:"100%",minHeight:0}}>
                  <div style={{flex:1,overflow:"auto",
                    borderRight:`1px solid ${D.border}`}}>
                    {/* Threat status */}
                    <div style={{padding:"7px 9px"}}>
                      <div style={{fontSize:7,color:D.textDim,letterSpacing:3,
                        fontFamily:D.FD,marginBottom:7}}>THREAT MATRIX</div>
                      {threats.map(t=>(
                        <div key={t.id} style={{marginBottom:6,padding:"6px 8px",
                          background:D.bg0,
                          border:`1px solid ${t.neutralised?D.g:t.detected?D.r:D.border}`,
                          borderRadius:3}}>
                          <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
                            <span style={{color:t.neutralised?D.g:t.detected?D.r:D.textDim,
                              fontSize:8.5,fontFamily:D.FD,letterSpacing:1}}>
                              THREAT-{t.id}
                            </span>
                            <span style={{fontSize:7.5,color:D.textDim,fontFamily:D.FM}}>
                              {t.type}
                            </span>
                          </div>
                          <div style={{fontSize:8,color:D.textDim,fontFamily:D.FM}}>
                            STATUS:{" "}
                            <span style={{color:t.neutralised?D.g:t.detected?D.r:D.y}}>
                              {t.neutralised?"NEUTRALISED":t.detected?"DETECTED":"UNDETECTED"}
                            </span>
                            {t.detectedBy!==null&&` · BY A${t.detectedBy}`}
                          </div>
                          <div style={{display:"flex",alignItems:"center",gap:4,marginTop:3}}>
                            <span style={{fontSize:7,color:D.textDim}}>THREAT LVL</span>
                            <div style={{width:60,height:3,background:D.muted,borderRadius:2,overflow:"hidden"}}>
                              <div style={{width:`${t.threatLevel*100}%`,height:"100%",background:D.r}}/>
                            </div>
                            <span style={{fontSize:7,color:D.r}}>{(t.threatLevel*100).toFixed(0)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{flex:1,overflow:"auto"}}>
                    <AgentDetailPanel agent={selAgent} agents={agents}/>
                  </div>
                </div>
              )}
              {tab==="POLICY"&&<PolicyTable agents={agents}/>}
              {tab==="COMMS"&&(
                <div>
                  <div style={{padding:"5px 8px",fontSize:7,color:D.textDim,
                    letterSpacing:2,fontFamily:D.FD,
                    borderBottom:`1px solid ${D.border}`}}>
                    ACOUSTIC CHANNEL LOG
                  </div>
                  <CommLogPanel log={commLog}/>
                  <div style={{padding:"5px 8px",borderTop:`1px solid ${D.border}`}}>
                    <div style={{fontSize:7,color:D.textDim,letterSpacing:2,
                      fontFamily:D.FD,marginBottom:6}}>EVENT LOG</div>
                    {events.slice(0,25).map((e,i)=>(
                      <div key={e.id||i} style={{fontSize:8,
                        color:{success:D.g,warn:D.y,sys:D.b,
                          info:D.textDim,danger:D.r}[e.kind]||D.textDim,
                        padding:"2px 0",
                        borderBottom:`1px solid ${D.border}18`,
                        fontFamily:D.FM}}>
                        <span style={{color:D.textDim,marginRight:6}}>[{e.tick}]</span>
                        {e.text}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {tab==="EMERGENT"&&(
                <EmergentBehaviourPanel agents={agents} state={sim}/>
              )}
            </div>
          </div>

          {/* Node strip */}
          <div style={{height:46,flexShrink:0,display:"flex",gap:4,
            alignItems:"center",padding:"4px 8px",background:D.bg0,
            border:`1px solid ${D.border}`,borderRadius:4}}>
            <span style={{fontSize:6.5,color:D.textDim,letterSpacing:2,
              marginRight:4,fontFamily:D.FD,flexShrink:0}}>
              NODES
            </span>
            {agents.map(a=>(
              <NodeBadge key={a.id} agent={a}
                selected={a.id===sel}
                onClick={()=>setSel(a.id===sel?null:a.id)}/>
            ))}
            <div style={{flex:1}}/>
            {/* RL config summary */}
            <div style={{display:"flex",gap:8,fontSize:7,
              color:D.textDim,fontFamily:D.FM}}>
              <span>γ={CFG.GAMMA}</span>
              <span>α_π={CFG.LR_ACTOR}</span>
              <span>α_Q={CFG.LR_CRITIC}</span>
              <span>B={CFG.COMM_BUDGET}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </>;
}

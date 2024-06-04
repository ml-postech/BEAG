import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import io
import networkx as nx
import torch
import random
from PIL import Image

class GraphPlanner:
    def __init__(self, args, low_replay, low_agent, high_agent, score, env):
        self.low_replay = low_replay
        self.low_agent = low_agent
        self.high_agent = high_agent
        self.env = env
        self.dim = args.subgoal_dim
        self.args = args
        self.score = score
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.graph = None
        self.deleted_node = []
        self.init_dist = 0
        self.n_graph_node = 0
        self.cutoff = args.cutoff
        self.wp_candi = None
        self.landmarks = None
        self.states = None
        self.waypoint_vec = None
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0
        self.edge_lengths = None
        self.edge_visit_counts = None
        self.initial_sample = args.initial_sample
        self.waypoint_bef_obs = None
        self.disconnected = []
        self.current = None
        self.n_succeeded_node = 0
        random.seed(self.args.seed)


    def fps_selection(
            self,
            landmarks,
            states,
            n_select: int,
            inf_value=1e6,
            low_bound_epsilon=10, early_stop=True,
    ):
        n_states = landmarks.shape[0]
        dists = np.zeros(n_states) + inf_value
        chosen = []
        while len(chosen) < n_select:
            if (np.max(dists) < low_bound_epsilon) and early_stop and (len(chosen) > self.args.n_graph_node/10):
                break
            idx = np.argmax(dists)  # farthest point idx
            farthest_state = states[idx]
            chosen.append(idx)
            # distance from the chosen point to all other pts
            if self.args.use_oracle_G:
                new_dists = self._get_dist_from_start_oracle(farthest_state, landmarks)
            else:
                new_dists = self.low_agent._get_dist_from_start(farthest_state, landmarks)
            new_dists[idx] = 0
            dists = np.minimum(dists, new_dists)
        return chosen
        
    def graph_construct(self, iter):
        if self.args.method == 'grid':
            self.current = None
            self.init_dist = self.args.init_dist
            if self.graph is None:
                if self.env.env_name == 'Reacher3D-v0':
                    replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                    landmarks = replay_data['ag']
                    x = np.arange(-1.0, 1.1, 0.4)
                    y = np.arange(-1.0, 1.1, 0.4)
                    z = np.arange(-1.0, 1.1, 0.4)
                    X,Y,Z = np.meshgrid(x, y, z)
                    self.landmarks = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
                    self.n_graph_node = self.landmarks.shape[0]
                    self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                    
                    self.graph = nx.DiGraph()
                    for i in range(self.n_graph_node):
                        for j in range(self.n_graph_node):
                            if i != j:
                                if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 0.5:
                                    self.graph.add_edge(i, j, weight = 1.)
                                    
                    nx.set_node_attributes(self.graph, 0, 'attempt_count')
                    nx.set_node_attributes(self.graph, 0, 'success_count')
                    nx.set_node_attributes(self.graph, 0, 'before')
                    nx.set_edge_attributes(self.graph, 0, 'visit_count')
                    l = landmarks.shape[0]
                    for i in range(l):
                        for j in range(self.n_graph_node):
                            dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                            if dist < 0.05:
                                self.graph.nodes[j]['success_count'] += 1       
                    return self.landmarks, self.states
                elif self.env.env_name == 'AntMaze':
                    self.xmin = -8
                    self.xmax = 24
                    self.ymin = -8
                    self.ymax = 24
                    replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                    landmarks = replay_data['ag']
                    self.landmarks = np.random.uniform(low=-4, high = 20, size = (800,2))
                    
                    # random
                    self.states = np.zeros((self.landmarks.shape[0], 29))
                    random_state = replay_data['ob'][0,2:29]
                    self.states[:,2:29] = random_state
                    self.states[:,:2] = self.landmarks
                    
                    self.n_graph_node = self.landmarks.shape[0]
                    self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                    
                    self.graph = nx.DiGraph()
                    for i in range(self.n_graph_node):
                        cnt = 0
                        for j in range(self.n_graph_node):
                            if i != j:
                                if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                    self.graph.add_edge(i, j, weight = 2.)
                                    cnt += 1
                        if cnt == 0:
                            self.graph.add_node(i)
                            
                    nx.set_node_attributes(self.graph, 2., 'distance')
                    nx.set_node_attributes(self.graph, 0, 'attempt_count')
                    nx.set_node_attributes(self.graph, 0, 'success_count')
                    nx.set_node_attributes(self.graph, 0, 'before')
                    nx.set_edge_attributes(self.graph, 0, 'visit_count')
                    
                    l = landmarks.shape[0]
                    for i in range(l):
                        for j in range(self.n_graph_node):
                            dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                            if dist < 0.5:
                                self.graph.nodes[j]['attempt_count'] += 1
                                self.graph.nodes[j]['success_count'] += 1       
                    for i in range(self.n_graph_node):
                        if self.graph.nodes[i]['success_count'] > 0:
                            self.n_succeeded_node += 1
                            
                    return self.landmarks, self.states
                
                elif self.env.env_name == 'AntMazeBottleneck':
                    self.xmin = -8
                    self.xmax = 24
                    self.ymin = -8
                    self.ymax = 24
                    replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                    landmarks = replay_data['ag']
                    x = np.arange(-5, 22, 2.0)
                    y = np.arange(-5, 22, 2.0)
                    X,Y = np.meshgrid(x, y)
                    self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                    self.n_graph_node = self.landmarks.shape[0]
                    self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                    
                    self.graph = nx.DiGraph()
                    for i in range(self.n_graph_node):
                        for j in range(self.n_graph_node):
                            if i != j:
                                if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                    self.graph.add_edge(i, j, weight = 2.)
                                    
                    nx.set_node_attributes(self.graph, 2., 'distance')
                    nx.set_node_attributes(self.graph, 0, 'attempt_count')
                    nx.set_node_attributes(self.graph, 0, 'success_count')
                    nx.set_node_attributes(self.graph, 0, 'before')
                    nx.set_edge_attributes(self.graph, 0, 'visit_count')
                    
                    l = landmarks.shape[0]
                    for i in range(l):
                        for j in range(self.n_graph_node):
                            dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                            if dist < 0.5:
                                self.graph.nodes[j]['attempt_count'] += 1
                                self.graph.nodes[j]['success_count'] += 1      
                                
                    return self.landmarks, self.states
                
                elif self.env.env_name == 'AntMazeP':
                    self.xmin = -13.5
                    self.xmax = 29.5
                    self.ymin = -5.5
                    self.ymax = 37.5
                    replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                    landmarks = replay_data['ag']
                    x = np.arange(-13, 30, 2.0)
                    y = np.arange(-5, 38, 2.0)
                    X,Y = np.meshgrid(x, y)
                    self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                    self.n_graph_node = self.landmarks.shape[0]
                    self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                    
                    self.graph = nx.DiGraph()
                    for i in range(self.n_graph_node):
                        for j in range(self.n_graph_node):
                            if i != j:
                                if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                    self.graph.add_edge(i, j, weight = 2.)
                                    
                    nx.set_node_attributes(self.graph, 2., 'distance')
                    nx.set_node_attributes(self.graph, 0, 'attempt_count')
                    nx.set_node_attributes(self.graph, 0, 'success_count')
                    nx.set_node_attributes(self.graph, 0, 'before')
                    nx.set_edge_attributes(self.graph, 0, 'visit_count')
                    
                    l = landmarks.shape[0]
                    for i in range(l):
                        for j in range(self.n_graph_node):
                            dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                            if dist < 0.5:
                                self.graph.nodes[j]['attempt_count'] += 1
                                self.graph.nodes[j]['success_count'] += 1       
                    return self.landmarks, self.states
                elif self.env.env_name == 'AntMazeComplex-v0':
                    self.xmin = -5.5
                    self.xmax = 54.5
                    self.ymin = -5.5
                    self.ymax = 54.5
                    replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
                    landmarks = replay_data['ag']
                    x = np.arange(-5, 54, 2.0)
                    y = np.arange(-5, 54, 2.0)
                    X,Y = np.meshgrid(x, y)
                    self.landmarks = np.array([X.flatten(), Y.flatten()]).T
                    self.n_graph_node = self.landmarks.shape[0]
                    self.edge_visit_counts = np.zeros((self.n_graph_node, self.n_graph_node))
                    
                    self.graph = nx.DiGraph()
                    for i in range(self.n_graph_node):
                        for j in range(self.n_graph_node):
                            if i != j:
                                if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) <= 2.02:
                                    self.graph.add_edge(i, j, weight = 2.)
                                    
                    nx.set_node_attributes(self.graph, 2., 'distance')
                    nx.set_node_attributes(self.graph, 0, 'attempt_count')
                    nx.set_node_attributes(self.graph, 0, 'success_count')
                    nx.set_node_attributes(self.graph, 0, 'before')
                    nx.set_edge_attributes(self.graph, 0, 'visit_count')
                    
                    l = landmarks.shape[0]
                    for i in range(l):
                        for j in range(self.n_graph_node):
                            dist = np.linalg.norm(landmarks[i]-self.landmarks[j])
                            if dist < 0.5:
                                self.graph.nodes[j]['attempt_count'] += 1
                                self.graph.nodes[j]['success_count'] += 1       
                    return self.landmarks, self.states
            else:
                return self.landmarks, self.states

            return self.landmarks, self.states

    def expand_node(self, anchor, dist):
        candi_list = []
        
        if self.dim == 2:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if i % 2 == 0 and j % 2 == 0:
                        continue
                    else:
                        if anchor[0]+i*dist <= self.xmin:
                            continue
                        if anchor[0]+i*dist >= self.xmax:
                            continue
                        if anchor[1]+j*dist <= self.ymin:
                            continue
                        if anchor[1]+j*dist >= self.ymax:
                            continue
                        candi_list.append([anchor[0]+i*dist, anchor[1]+j*dist, dist])
        elif self.dim == 3:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                            continue
                        else:
                            candi_list.append([anchor[0]+i*dist, anchor[1]+j*dist, anchor[2]+k*dist, dist])
    
        return np.array(candi_list)
            
    def expand(self):
        edges = self.graph.edges(data=True)
        node_type = np.zeros(self.landmarks.shape[0])
        # 0 : untried, 1 : success, 2 : failed
        for i in range(self.landmarks.shape[0]):
            if self.graph.nodes[i]['attempt_count'] == 0:
                node_type[i] = 0
            elif self.graph.nodes[i]['success_count'] != 0:
                node_type[i] = 1
            else:
                node_type[i] = 2
        
        for edge in edges:
            if (node_type[edge[0]] + node_type[edge[1]] == 1):
                return False
        
        distance_reduce = np.zeros(self.landmarks.shape[0])
        removed_edges = []
        
        for edge in edges:
            if (node_type[edge[0]] == 2) or (node_type[edge[1]] == 2):
                distance_reduce[edge[0]] = 1
                distance_reduce[edge[1]] = 1
                
        for edge in edges:
            if (distance_reduce[edge[0]]) and (distance_reduce[edge[1]]):
                removed_edges.append((edge[0], edge[1]))
                
        for edge in removed_edges:
            self.graph.remove_edge(edge[0], edge[1])
        landmark_candi = []
        for i in range(self.landmarks.shape[0]):
            if distance_reduce[i] == 1:
                self.graph.nodes[i]['distance'] = self.graph.nodes[i]['distance'] / 2.
            if node_type[i] == 2:
                anchor = self.landmarks[i]
                dist = self.graph.nodes[i]['distance']
                self.graph.nodes[i]['attempt_count'] = 0
                candi_list = self.expand_node(anchor, dist)
                if len(landmark_candi) == 0:
                    landmark_candi = candi_list
                else:
                    landmark_candi = np.concatenate((landmark_candi, candi_list))
        
        if len(landmark_candi) == 0:
            return False

        landmark_candi = np.unique(landmark_candi, axis = 0)
        n = self.n_graph_node
        self.landmarks = np.concatenate((self.landmarks, landmark_candi[:,:2]))
        self.n_graph_node = self.landmarks.shape[0]
        
        for i in range(n, self.n_graph_node):
            dist_ = landmark_candi[i-n, 2]
            self.graph.add_node(i)
            self.graph.nodes[i]['attempt_count'] = 0
            self.graph.nodes[i]['success_count'] = 0
            self.graph.nodes[i]['before'] = 0
            self.graph.nodes[i]['distance'] = dist_
            for j in range(i):
                dist = np.linalg.norm(self.landmarks[i]-self.landmarks[j])
                threshold = np.max([dist_, self.graph.nodes[j]['distance']])
                if (dist <= threshold * 1.01):
                    self.graph.add_edge(i, j, weight = dist)#threshold)
                    self.graph.add_edge(j, i, weight = dist)#threshold)
        
        self.disconnected = []
        
        return True
    
    def dense(self, dg):
        edges = self.graph.edges(data=True)
        
        remove_edges = []
        for edge in edges:
            if edge[0] == dg or edge[1] == dg:
                remove_edges.append(edge)
                
        for edge in remove_edges:
            self.graph.remove_edge(*edge[:2])
        
        self.graph.nodes[dg]['attempt_count'] = 0
        self.graph.nodes[dg]['success_count'] = 0
        dist = self.graph.nodes[dg]['distance'] / 2.
        self.graph.nodes[dg]['distance'] = dist
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                exist = False
                candi = np.array([0., 0.])
                candi[0] = self.landmarks[dg][0] + dist * i
                candi[1] = self.landmarks[dg][1] + dist * j
                for k in range(self.n_graph_node):
                    if np.linalg.norm(self.landmarks[k] - candi) < 0.01:
                        exist = True
                        self.graph.nodes[k]['attempt_count'] = 0
                        self.graph.nodes[k]['success_count'] = 0
                        for l in range(self.n_graph_node):
                            if l != k:
                                d = np.max([dist, self.graph.nodes[l]['distance']])
                                if np.linalg.norm(self.landmarks[k] - self.landmarks[l]) < 1.01 * d:
                                   
                                    if self.graph.has_edge(k, l):
                                        self.graph[k][l]['weight'] = np.linalg.norm(self.landmarks[k] - self.landmarks[l])
                                        self.graph[l][k]['weight'] = np.linalg.norm(self.landmarks[k] - self.landmarks[l])
                                        self.graph.nodes[k]['distance'] = d
                        if k in self.disconnected:
                            self.disconnected.remove(k)
                        
                if not exist:
                    candi = np.expand_dims(candi, axis = 0)
                    self.landmarks = np.concatenate((self.landmarks, candi))
                    self.graph.add_node(self.n_graph_node)
                    self.graph.nodes[self.n_graph_node]['attempt_count'] = 0
                    self.graph.nodes[self.n_graph_node]['success_count'] = 0
                    self.graph.nodes[self.n_graph_node]['distance'] = dist
                    for m in range(self.n_graph_node):
                        d = np.max([dist,self.graph.nodes[m]['distance']])
                        if np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]) < 1.01 * d:
                            if((self.landmarks[self.n_graph_node][0] == self.landmarks[m][0]) or (self.landmarks[self.n_graph_node][1] == self.landmarks[m][1])):
                                self.graph.add_edge(m, self.n_graph_node, weight = np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]))
                                self.graph.add_edge(self.n_graph_node, m, weight = np.linalg.norm(self.landmarks[self.n_graph_node] - self.landmarks[m]))
                    
                    self.n_graph_node += 1
                    
    
    def densify(self):
        failed = []
        cnt = 0
        for i in range(self.n_graph_node):
            if(self.graph.nodes[i]['success_count'] > 0):
                cnt += 1
        for i in range(self.n_graph_node):
            if((self.graph.nodes[i]['attempt_count'] > 3) and (self.graph.nodes[i]['success_count'] == 0)):
                failed.append(self.graph.nodes[i]['distance'])
            else:
                failed.append(0)
                
        failed = np.array(failed)
        max_dist = np.max(failed)
        
        if cnt > self.n_succeeded_node:
            self.n_succeeded_node = cnt
            return
        if max_dist == 0:
            return
        candi = np.where(failed == max_dist)
        
        self.dense(candi[0][random.choices(range(len(candi[0])))][0])
        
        return 
    
    def find_path(self, ob, subgoal, ag, bg, inf_value=1e6, train = False, first = False):
        expanded_graph = self.graph.copy()
        self.edge_lengths = []
        if self.args.nosubgoal:
            subgoal = bg#subgoal[:self.dim]
        else:
            subgoal = subgoal[:self.dim]    
        self.wp_candi = None
        
        if self.args.method == 'grid':
            if first:
                self.deleted_node = []
            if self.graph is not None:
                if self.expand():
                    expanded_graph = self.graph.copy()
            if self.deleted_node:
                for i in self.deleted_node:
                    for j in range(self.n_graph_node):
                        if i != j:
                            threshold = np.max([expanded_graph.nodes[i]['distance'], expanded_graph.nodes[j]['distance']])
                            if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) < threshold * 1.01:
                                if expanded_graph.has_edge(i, j):
                                    expanded_graph[i][j]['weight'] = 1000.
                                    expanded_graph[j][i]['weight'] = 1000.
            
            start_to_goal_length = np.linalg.norm(ag - bg)
            if start_to_goal_length < 2.0:
                expanded_graph.add_edge('start', 'goal', weight = 1.)
                
            start_edge_length = self.dist_to_graph(ag, self.landmarks)
            goal_edge_length = self.dist_to_graph(bg, self.landmarks)
            
            self.edge_lengths = [] 
            
            for i in range(self.n_graph_node):
                if start_edge_length[i] < 2.0:
                    if i not in self.disconnected:
                        expanded_graph.add_edge('start', i, weight = 1.)
                    else:
                        expanded_graph.add_edge('start', i, weight = 1000.)
                if goal_edge_length[i] < 2.0:
                    if i not in self.disconnected:
                        expanded_graph.add_edge(i, 'goal', weight = 1.)
                    else:
                        expanded_graph.add_edge(i, 'goal', weight = 1000.)
            if (not expanded_graph.has_node('start')):
                added = False
                adjusted = 1.5
                while True:
                    adjusted_cutoff = 2.0 * adjusted
                    for i in range(self.n_graph_node):
                        if(start_edge_length[i] < adjusted_cutoff):
                            if i not in self.disconnected:
                                expanded_graph.add_edge('start', i, weight = 1.)
                                added = True
                    if added:
                        break
                    adjusted += 0.5           
            
            if(not expanded_graph.has_node('goal')):
                adjusted_cutoff = 2.0 * 2.0
                for i in range(self.n_graph_node):
                    if(goal_edge_length[i] < adjusted_cutoff):
                        if i not in self.disconnected:
                            expanded_graph.add_edge(i, 'goal', weight = 1.)
            
            if(not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal')):
                while True:
                    nearestnode = np.argmin(goal_edge_length) #nearest point from the goal
                    if goal_edge_length[nearestnode] > start_to_goal_length:
                        expanded_graph.add_edge('start', 'goal', weight = 1.)
                        break
                    if(expanded_graph.has_node(nearestnode)) and (nx.has_path(expanded_graph, 'start', nearestnode)):
                        expanded_graph.add_edge(nearestnode, 'goal', weight = goal_edge_length[nearestnode])
                        break
                    else:
                        goal_edge_length[nearestnode] = inf_value
                        
            path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
            for (i, j) in zip(path[:-1], path[1:]):
                self.edge_lengths.append(expanded_graph[i][j]['weight'])
                
            self.waypoint_vec = list(path)[1:-1]
            self.waypoint_idx = 0
            self.waypoint_chase_step = 0
            self.wp_candi = subgoal
            
            return self.wp_candi

    def get_goal_candi(self, expanded_graph):
        start_edge_length = []
        exist = False
        for i in range(self.n_graph_node):
            if self.graph.nodes[i]['attempt_count'] == 0:
                if nx.has_path(expanded_graph, 'start', i):
                    start_edge_length.append(nx.shortest_path_length(expanded_graph, source='start', target=i, weight='weight'))
                    exist = True
                else:
                    start_edge_length.append(5e3)
            else:
                start_edge_length.append(5e3)
        if exist:
            return self.landmarks[np.argmin(start_edge_length)]
        return None
            
    def check_easy_goal(self, ob, ag, subgoal):
        expanded_graph = self.graph.copy()
        subgoal = subgoal[:self.dim]
        
        goal_edge_length = self.dist_to_graph(subgoal, self.landmarks)
        for i in range(self.n_graph_node):
            if goal_edge_length[i] < 2.02:
                if self.graph.nodes[i]['success_count'] > 0:
                    if self.deleted_node:
                        for i in self.deleted_node:
                            for j in range(self.n_graph_node):
                                if i != j:
                                    threshold = np.max([expanded_graph.nodes[i]['distance'], expanded_graph.nodes[j]['distance']])
                                    if np.linalg.norm(self.landmarks[i] - self.landmarks[j]) < threshold * 1.01:
                                        if expanded_graph.has_edge(i, j):
                                            expanded_graph[i][j]['weight'] = 1000.
                                            expanded_graph[j][i]['weight'] = 1000.
                    start_to_goal_length = np.linalg.norm(ag - subgoal)
                    if start_to_goal_length < 2.0:
                        expanded_graph.add_edge('start', 'goal', weight = 1.)
                        
                    start_edge_length = self.dist_to_graph(ag, self.landmarks)
                    goal_edge_length = self.dist_to_graph(subgoal, self.landmarks)
                    
                    self.edge_lengths = [] 
                    
                    for i in range(self.n_graph_node):
                        if start_edge_length[i] < 2.0:
                            if i not in self.disconnected:
                                expanded_graph.add_edge('start', i, weight = 1.)
                            else:
                                expanded_graph.add_edge('start', i, weight = 1000.)
                        if goal_edge_length[i] < 2.0:
                            if i not in self.disconnected:
                                expanded_graph.add_edge(i, 'goal', weight = 1.)
                            else:
                                expanded_graph.add_edge(i, 'goal', weight = 1000.)
                    if (not expanded_graph.has_node('start')):
                        added = False
                        adjusted = 1.5
                        while True:
                            adjusted_cutoff = 2.0 * adjusted
                            for i in range(self.n_graph_node):
                                if(start_edge_length[i] < adjusted_cutoff):
                                    if i not in self.disconnected:
                                        expanded_graph.add_edge('start', i, weight = 1.)
                                        added = True
                            if added:
                                break
                            adjusted += 0.5
                    return self.get_goal_candi(expanded_graph)
            return None
    
    def dist_from_graph_to_goal(self, subgoal):
        dist_list=[]
        for i in range(subgoal.shape[0]):  
            curr_subgoal = subgoal[i,:self.dim]
            if self.args.use_oracle_G:
                goal_edge_length = self._get_dist_to_goal_oracle(self.states, curr_subgoal)
            else:
                goal_edge_length = self.low_agent._get_dist_to_goal(self.states, curr_subgoal)
            dist_list.append(min(goal_edge_length))
        return np.array(dist_list)
    
    def dist_to_graph(self, node, landmarks):
        return np.linalg.norm(node[:self.dim]-landmarks, axis = 1)
            
    
    def get_waypoint(self, ob, ag, subgoal, bg, train=False):
        if self.graph is not None:
            if self.args.method == 'grid':
                self.waypoint_chase_step += 1
                if(self.waypoint_idx >= len(self.waypoint_vec)):
                    waypoint_subgoal = bg
                else:
                    i = self.waypoint_vec[self.waypoint_idx]
                
                    if((np.linalg.norm(ag[:self.dim]-self.landmarks[i][:self.dim]) < 0.5)):

                        if train:
                            self.graph.nodes[i]['attempt_count'] += 1
                            self.graph.nodes[i]['success_count'] += 1
                            self.graph.nodes[i]['before'] = 0
                        
                        self.waypoint_idx += 1
                        self.waypoint_chase_step = 0
                        
                    elif((self.waypoint_chase_step > 100.)):
                        if train:
                            self.graph.nodes[i]['attempt_count'] += 1
                        if self.graph.nodes[i]['success_count'] == 0:
                            if train:
                                if self.graph.nodes[i]['attempt_count'] > 3:
                                    self.disconnected.append(i)
                                    for j in range(self.n_graph_node):
                                        if i != j:
                                            if self.graph.has_edge(i, j):
                                                self.graph[i][j]['weight'] = 1000.
                                                self.graph[j][i]['weight'] = 1000.
                                                    
                                    self.find_path(ob, subgoal, ag, bg)
                                else:
                                    self.deleted_node.append(i)
                                    self.find_path(ob, subgoal, ag, bg)
                            else:
                                self.deleted_node.append(i)
                                self.find_path(ob, subgoal, ag, bg)
                        else:
                            self.deleted_node.append(i)
                            self.find_path(ob, subgoal, ag, bg)
                            
                    if(self.waypoint_idx >= len(self.waypoint_vec)):
                        waypoint_subgoal = bg
                    else:
                        waypoint_subgoal = self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]
            
        else:
            waypoint_subgoal = subgoal
        return waypoint_subgoal

    
    def draw_edge_graph(self):
        plt.cla()
        new_graph = nx.Graph()
        new_graph.add_edges_from((u,v,d) for u,v,d in self.graph.edges(data=True) if d['visit_count'] > 0)
        # edge_colors = [self.graph[edge[0]][edge[1]]['visit_count'] for edge in self.graph.edges()]
        edge_colors = [new_graph[edge[0]][edge[1]]['visit_count'] for edge in new_graph.edges()]
        # print(np.max(edge_colors))
        pos = {idx: (landmark[0], landmark[1]) for idx, landmark in enumerate(self.landmarks)}
        nx.draw_networkx_nodes(self.graph, pos, node_size=5, node_color='k')
        edge_colloection = nx.draw_networkx_edges(new_graph, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=1, arrows=False, style='-')
        plt.colorbar(edge_colloection)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    def draw_graph(self, start=None, subgoal=None, goal=None):
        map_size = [-4, 20]
        wall_x = None
        wall_y = None
        if self.env.env_name == 'AntMaze':
            # -4~ 20
            map_size = [-6, 22]
            wall_x = [-4, 20, 20, -4, -4, 12, 12, -4, -4]
            wall_y = [-4, -4, 20, 20, 12, 12, 4, 4 , -4]
            Map_x, Map_y = (24, 24)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeBottleneck':
            map_size = [-8, 24]
            wall_x = [-4, 20, 20, 17, 17, 20, 20, -4, -4, 12, 12, 15, 15, 12, 12, -4, -4]
            wall_y = [-4, -4,  7,  7,  9,  9, 20, 20, 12, 12,  9,  9,  7,  7,  4,  4, -4]
            Map_x, Map_y = (24, 24)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeMultiPathBottleneck':
            map_size = [-8, 24]
            wall_x = [ 4,  4, 15, 15, 4]
            wall_y = [ 4, 12, 12, 4, 4]
            wall_x2 = [17, 20, 20, -4, -4, 20, 20, 17, 17]
            wall_y2 = [12, 12, 20, 20, -4, -4, 4, 4, 12]
        elif self.env.env_name == 'AntMazeSmall-v0':
            # -2 ~ 12
            map_size = [-2, 12]
            Map_x, Map_y = (12, 12)
            start_x, start_y = (2,2)
        elif self.env.env_name == 'AntMazeS':
            map_size = [-6, 38]
            wall_x = [-4, 36, 36,  4,  4, 36, 36, -4, -4, 28, 28, -4, -4]
            wall_y = [-4, -4, 20, 20, 28, 28, 36, 36, 12, 12,  4,  4, -4]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 4)
        elif self.env.env_name == 'AntMazeW':
            map_size_x = [-6, 38]
            map_size_y = [-14, 30]
            wall_x = [ -4,  36, 36, -4, -4,  4,  4, 28, 28, 12, 12, 28, 28,  4,  4, -4, -4]
            wall_y = [-12, -12, 28, 28, 12, 12, 20, 20, 12, 12,  4,  4, -4, -4,  4,  4, -12]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (4, 12)
        elif self.env.env_name == 'AntMazeP':
            map_size_x = [-16, 32]
            map_size_y = [-8, 40]
            wall_x = [-12,  4,  4, -4, -4,  4,  4, 12, 12, 20, 20, 12, 12, 28, 28, 20, 20, 28, 28, -12, -12, -4, -4, -12, -12]
            wall_y = [ -4, -4,  4,  4, 12, 12, 28, 28, 12, 12,  4,  4, -4, -4, 20, 20, 28, 28, 36,  36,  28, 28, 20,  20,  -4]
            Map_x, Map_y = (40, 40)
            start_x, start_y = (12, 4)
        elif self.env.env_name == 'AntMazeMultiPath-v0':
            # -2 ~ 12
            map_size = [-2, 12]
            Map_x, Map_y = (12, 12)
            start_x, start_y = (6,2)    
        elif self.env.env_name == 'AntMazeComplex-v0':
            # -4 ~ 52
            map_size = [-4, 52]
            wall_x = [-4, -4, 12, 12, -4, -4, 4, 4, 12, 12, 20, 20, 28, 28, 52, 52, 44, 44, 36, 36, 44, 44, 52, 52, 28, 28, 36, 36, 28, 28, 20, 20, 12, 12, 4, 4, 20, 20, -4]
            wall_y = [-4, 4, 4, 12, 12, 52, 52, 44, 44, 52, 52, 44, 44, 52, 52, 36, 36, 44, 44, 28, 28, 12, 12, -4, -4, 12, 12, 20, 20, 36, 36, 28, 28, 36, 36, 20, 20, -4, -4]
            Map_x, Map_y = (56, 56)
            start_x, start_y = (4, 4)
        # First Graph
        fig1, ax1 = plt.subplots()
        if self.env.env_name == 'AntMazeP' or self.env.env_name == 'AntMazeW':
            ax1.set_xlim(map_size_x)
            ax1.set_ylim(map_size_y)
        else:
            ax1.set_xlim(map_size)
            ax1.set_ylim(map_size)
        x_vertex = []
        y_vertex = []
        for landmark in self.landmarks:
            x_vertex.append(landmark[0])
            y_vertex.append(landmark[1])

        edges = self.graph.edges(data=True)
        x_edges = []
        y_edges = []
        for edge in edges:
            node1, node2 = edge[0], edge[1]
            if self.graph[node1][node2]['weight'] < 100.:
                # if node1 < node2:
                x_edges.append((self.landmarks[node1][0], self.landmarks[node2][0]))
                y_edges.append((self.landmarks[node1][1], self.landmarks[node2][1]))
        
        x_edges = np.array(x_edges)
        y_edges = np.array(y_edges)
        
        ax1.scatter(x_vertex, y_vertex, c='k', marker='o', alpha=1)
        ax1.plot(x_edges.T, y_edges.T, c='k', alpha=0.2)
        
        ax1.plot(wall_x, wall_y, c ='k')
        
        if self.env.env_name == 'AntMazeMultiPathBottleneck':
            ax1.plot(wall_x2, wall_y2, c='k')
        
        buf1 = io.BytesIO()
        
        fig1.savefig(buf1, format='png')
        buf1.seek(0)
        image1 = Image.open(buf1)
        numpy_array1= np.array(image1)
        plt.close()
        # Second Graph
        fig, ax = plt.subplots()
        if self.env.env_name == 'AntMazeP' or self.env.env_name == 'AntMazeW':
            ax.set_xlim(map_size_x)
            ax.set_ylim(map_size_y)
        else:
            ax.set_xlim(map_size)
            ax.set_ylim(map_size)
        x_waypoint = []
        y_waypoint = []
        x_waypoint_edges = []
        y_waypoint_edges = []
 
        if start is not None and subgoal is not None:
            bef_x_waypoint = start[0]
            bef_y_waypoint = start[1]

            for waypoint_idx in self.waypoint_vec:
                if waypoint_idx < self.n_graph_node:
                    waypoint_subgoal = self.landmarks[waypoint_idx][:self.dim]
                    x_waypoint.append(waypoint_subgoal[0])
                    y_waypoint.append(waypoint_subgoal[1])
                    x_waypoint_edges.append((bef_x_waypoint, waypoint_subgoal[0]))
                    y_waypoint_edges.append((bef_y_waypoint, waypoint_subgoal[1]))
                    bef_x_waypoint = waypoint_subgoal[0]
                    bef_y_waypoint = waypoint_subgoal[1]

            x_waypoint_edges.append((bef_x_waypoint, subgoal[0]))
            y_waypoint_edges.append((bef_y_waypoint, subgoal[1]))
            ax.plot(x_waypoint_edges, y_waypoint_edges, c='k', alpha=1)
            ax.scatter(x_waypoint, y_waypoint, c='g', marker='o')   
            ax.scatter(start[0], start[1], c='r', marker='o')
            ax.scatter(subgoal[0], subgoal[1], c='b', marker='o')

        if goal is not None:
            x_goal, y_goal = goal[0], goal[1]
            ax.scatter([x_goal], [y_goal], c='k', marker='o', alpha=1)
        ax.scatter(x_vertex, y_vertex, c='k', marker='o', alpha=0.1)
        ax.plot(x_edges.T, y_edges.T, c='k', alpha=0.1)
        if wall_x is not None:
            ax.plot(wall_x, wall_y, c ='k')
        if self.env.env_name == 'AntMazeMultiPathBottleneck':
            ax.plot(wall_x2, wall_y2, c='k')
            
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        numpy_array = np.array(image)
        plt.close()
        return numpy_array, numpy_array1

    #####################oracle graph#########################
    def _get_dist_to_goal_oracle(self, obs_tensor, goal):
        goal_repeat = np.ones_like(obs_tensor[:, :self.args.subgoal_dim]) \
            * np.expand_dims(goal[:self.args.subgoal_dim], axis=0)
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - goal_repeat, axis=1)
        return dist

    def _get_dist_from_start_oracle(self, start, obs_tensor):
        start_repeat = np.ones_like(obs_tensor) * np.expand_dims(start, axis=0)
        start_repeat = start_repeat[:, :self.args.subgoal_dim]
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - start_repeat, axis=1)
        return dist

    def _get_point_to_point_oracle(self, point1, point2):
        point1 = point1[:self.args.subgoal_dim]
        point2 = point2[:self.args.subgoal_dim]
        dist = np.linalg.norm(point1-point2)
        return dist

    def _get_pairwise_dist_oracle(self, obs_tensor):
        goal_tensor = obs_tensor
        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
            obs = obs_tensor[obs_index]
            obs_repeat_tensor = np.ones_like(goal_tensor) * np.expand_dims(obs, axis=0)
            dist = np.linalg.norm(obs_repeat_tensor[:, :self.args.subgoal_dim] - goal_tensor[:, :self.args.subgoal_dim], axis=1)
            dist_matrix.append(np.squeeze(dist))
        pairwise_dist = np.array(dist_matrix) #pairwise_dist[i][j] is dist from i to j
        return pairwise_dist
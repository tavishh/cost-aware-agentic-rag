"""RL environment over the cached tree15 files.

Step 2.2: actions and masking.
Step 2.3: transitions (TreeEnv.reset / TreeEnv.step).
"""
from common.schemas import Action
from rl.tree_data import child_id

# Fixed order: index i of the DQN output always means ACTIONS[i].
ACTIONS = (Action.ANSWER, Action.RETRIEVE, Action.REWRITE, Action.ABSTAIN)
TERMINAL = (Action.ANSWER, Action.ABSTAIN)


def valid_actions(nodes, node_id):
    """Actions allowed at node_id. RETRIEVE/REWRITE need the child to exist in the cache."""
    allowed = []
    for a in ACTIONS:
        if a in TERMINAL or child_id(node_id, a.value) in nodes:
            allowed.append(a)
    return allowed


def action_mask(nodes, node_id):
    """Boolean mask in ACTIONS order, used to block invalid actions in the DQN."""
    allowed = valid_actions(nodes, node_id)
    return [a in allowed for a in ACTIONS]


class TreeEnv:
    """One episode = one question. The state is the current node in that question's tree."""

    def __init__(self, nodes, roots):
        self.nodes, self.roots = nodes, roots
        self.node_id, self.path, self.done, self.final_action = None, [], True, None

    def reset(self, qid):
        """Start an episode at the root node of question qid."""
        self.node_id = self.roots[qid]
        self.path = [self.node_id]          # every node visited (one LLM generation each)
        self.done, self.final_action = False, None
        return self.node_id

    def step(self, action):
        """Apply action. Returns (node_id, done); the reward is added in Step 2.4."""
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if action not in valid_actions(self.nodes, self.node_id):
            raise ValueError(f"{action} is not allowed at {self.node_id}")
        if action in TERMINAL:
            self.done, self.final_action = True, action
        else:
            self.node_id = child_id(self.node_id, action.value)
            self.path.append(self.node_id)
        return self.node_id, self.done

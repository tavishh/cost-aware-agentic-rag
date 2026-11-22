"""RL environment over the cached tree15 files.

Step 2.2: actions and masking.
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
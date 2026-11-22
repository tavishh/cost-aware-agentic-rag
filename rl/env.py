"""RL environment over the cached tree15 files.

Step 2.2: actions and masking.
Step 2.3: transitions (TreeEnv.reset / TreeEnv.step).
Step 2.4: reward and cost.
"""
import re
import string

from common.schemas import Action
from rl.tree_data import child_id

# Fixed order: index i of the DQN output always means ACTIONS[i].
ACTIONS = (Action.ANSWER, Action.RETRIEVE, Action.REWRITE, Action.ABSTAIN)
TERMINAL = (Action.ANSWER, Action.ABSTAIN)


def valid_actions(nodes, node_id, allow_abstain=True):
    """Actions allowed at node_id. RETRIEVE/REWRITE need the child to exist in the cache.
    allow_abstain=False removes ABSTAIN (used for the no-abstention accuracy-cost frontier)."""
    allowed = []
    for a in ACTIONS:
        if a is Action.ABSTAIN and not allow_abstain:
            continue
        if a in TERMINAL or child_id(node_id, a.value) in nodes:
            allowed.append(a)
    return allowed


def action_mask(nodes, node_id, allow_abstain=True):
    """Boolean mask in ACTIONS order, used to block invalid actions in the DQN."""
    allowed = valid_actions(nodes, node_id, allow_abstain)
    return [a in allowed for a in ACTIONS]


def normalize(s):
    """SQuAD-style normalization: lowercase, drop punctuation and articles, collapse spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def inacc(pred, golds, strict=True):
    """1 if a gold answer appears in pred. strict=True requires whole-word matches
    ("1" does not match "1990"); strict=False is the standard substring InAcc."""
    p = normalize(pred)
    for g in golds:
        g = normalize(g)
        if g and ((f" {g} " in f" {p} ") if strict else (g in p)):
            return 1
    return 0


class TreeEnv:
    """One episode = one question. The state is the current node in that question's tree."""

    def __init__(self, nodes, roots, lam=1.0, c_ret=0.05, c_llm=0.02, strict=True,
                 allow_abstain=True):
        self.nodes, self.roots = nodes, roots
        self.allow_abstain = allow_abstain
        self.lam, self.c_ret, self.c_llm, self.strict = lam, c_ret, c_llm, strict
        self.node_id, self.path, self.done, self.final_action = None, [], True, None

    def reset(self, qid):
        """Start an episode at the root node of question qid."""
        self.node_id = self.roots[qid]
        self.path = [self.node_id]          # every node visited (one LLM generation each)
        self.done, self.final_action = False, None
        return self.node_id

    def step(self, action):
        """Apply action. Returns (node_id, reward, done, info)."""
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if action not in valid_actions(self.nodes, self.node_id, self.allow_abstain):
            raise ValueError(f"{action} is not allowed at {self.node_id}")

        if action is Action.ANSWER:
            node = self.nodes[self.node_id]
            gold = self.nodes[self.path[0]]["gold"]
            correct = inacc(node["answer_call"]["text"], gold, self.strict)
            reward = 1.0 if correct else -self.lam
            self.done, self.final_action = True, action
            info = self.episode_info(correct)
            # Evaluation only: literature-style substring InAcc of the same answer.
            info["correct_substring"] = inacc(node["answer_call"]["text"], gold, strict=False)
            return self.node_id, reward, True, info

        if action is Action.ABSTAIN:
            self.done, self.final_action = True, action
            info = self.episode_info(None)
            info["correct_substring"] = None
            return self.node_id, 0.0, True, info

        # RETRIEVE or REWRITE: move to the child, pay for the retrieval and the new draft.
        reward = -self.c_ret - self.c_llm
        if action is Action.REWRITE:
            reward -= self.c_llm                      # the extra LLM call that wrote the query
        self.node_id = child_id(self.node_id, action.value)
        self.path.append(self.node_id)
        return self.node_id, reward, False, {}

    def episode_info(self, correct):
        """Cost metrics for the finished episode. correct is None when the agent abstained."""
        visited = [self.nodes[n] for n in self.path]
        rewrites = sum(1 for n in visited if n["action"] == "rewrite")
        tokens = sum(n["answer_call"]["tokens_in"] + n["answer_call"]["tokens_out"] for n in visited)
        tokens += sum(n["rewrite_call"]["tokens_in"] + n["rewrite_call"]["tokens_out"]
                      for n in visited if "rewrite_call" in n)
        return {"correct": correct,
                "abstained": correct is None,
                "retrieval_calls": len(self.path) - 1,
                "llm_calls": len(self.path) + rewrites,   # one draft per node + each rewrite
                "tokens": tokens,
                "end_node": self.node_id}

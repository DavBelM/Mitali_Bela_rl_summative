import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Topics covered in Rwanda TVET Level 3 JavaScript curriculum
TOPICS = [
    "variables_and_datatypes",
    "control_flow",
    "functions",
    "arrays",
    "objects",
    "dom_manipulation",
    "events",
    "async_callbacks",
]

# Tutor actions the agent can take
ACTIONS = {
    0: "explain_concept",
    1: "give_easy_exercise",
    2: "give_hard_exercise",
    3: "provide_hint",
    4: "give_feedback",
    5: "reduce_difficulty",
    6: "advance_topic",
    7: "run_assessment",
}

NUM_TOPICS = len(TOPICS)
NUM_ACTIONS = len(ACTIONS)


class EduCodeEnv(gym.Env):
    """
    A simulated student learning environment based on the EduCode Rwanda platform.

    The agent plays the role of an AI tutor. It observes the current state of a
    simulated student and decides what instructional action to take. The goal is
    to guide the student through all JavaScript topics while keeping engagement
    high and frustration low.

    Observation space (9 continuous values, all normalized to [0, 1]):
        - current_topic_index     : which topic the student is on (normalized)
        - mastery                 : how well the student knows the current topic
        - error_rate              : recent error rate on exercises (0 = perfect)
        - engagement              : student engagement / motivation level
        - consecutive_correct     : streak of correct answers (normalized)
        - consecutive_wrong       : streak of wrong answers (normalized)
        - time_on_topic           : how long the student has spent on this topic
        - difficulty_level        : current exercise difficulty (0=easy, 1=hard)
        - topics_completed        : fraction of total topics mastered so far

    Action space (discrete, 8 actions):
        See ACTIONS dict above.

    Reward structure:
        - Mastery improves         : +2.0
        - Engagement improves      : +0.5
        - Student advances topic   : +5.0
        - All topics completed     : +10.0
        - Frustration (wrong x3+)  : -1.5
        - Giving hard exercise when struggling : -1.0
        - Doing nothing useful     : -0.3 (time penalty)
        - Student drops out        : -8.0

    Terminal conditions:
        - Student masters all topics (success)
        - Student drops out due to frustration (failure)
        - Max steps reached (timeout)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, max_steps=300, student_profile="average"):
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.student_profile = student_profile

        # Student profile adjusts how quickly mastery builds and how easily
        # they get frustrated. Profiles: "struggling", "average", "advanced"
        self._profile_params = {
            "struggling": {"learn_rate": 0.06, "frustration_thresh": 2, "engage_decay": 0.04},
            "average":    {"learn_rate": 0.10, "frustration_thresh": 3, "engage_decay": 0.02},
            "advanced":   {"learn_rate": 0.16, "frustration_thresh": 4, "engage_decay": 0.01},
        }
        self.profile = self._profile_params.get(student_profile, self._profile_params["average"])

        low  = np.zeros(9, dtype=np.float32)
        high = np.ones(9,  dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.renderer = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_topic     = 0
        self.mastery           = np.zeros(NUM_TOPICS, dtype=np.float32)
        self.error_rate        = 0.5
        self.engagement        = 0.7 + self.np_random.uniform(-0.1, 0.1)
        self.consec_correct    = 0
        self.consec_wrong      = 0
        self.time_on_topic     = 0
        self.difficulty        = 0.3
        self.topics_completed  = 0
        self.steps             = 0
        self.done              = False
        self.last_action       = None
        self.last_reward       = 0.0
        self.history           = []

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        obs = np.array([
            self.current_topic / (NUM_TOPICS - 1),
            self.mastery[self.current_topic],
            self.error_rate,
            self.engagement,
            min(self.consec_correct / 5.0, 1.0),
            min(self.consec_wrong   / 5.0, 1.0),
            min(self.time_on_topic  / 20.0, 1.0),
            self.difficulty,
            self.topics_completed / NUM_TOPICS,
        ], dtype=np.float32)
        return obs

    def _simulate_student_response(self, action):
        """
        Simulate how the student responds to the tutor's action.
        Returns (mastery_delta, engagement_delta, got_correct, got_wrong).
        """
        lr = self.profile["learn_rate"]
        mastery_now = self.mastery[self.current_topic]

        mastery_delta    = 0.0
        engagement_delta = 0.0
        got_correct      = False
        got_wrong        = False

        if action == 0:  # explain_concept
            mastery_delta    = lr * 0.8 * (1 - mastery_now)
            engagement_delta = 0.02 if mastery_now < 0.4 else -0.01

        elif action == 1:  # give_easy_exercise
            success_prob = 0.5 + mastery_now * 0.5
            if self.np_random.random() < success_prob:
                mastery_delta    = lr * 0.6
                engagement_delta = 0.04
                got_correct      = True
            else:
                mastery_delta    = lr * 0.1
                engagement_delta = -0.02
                got_wrong        = True

        elif action == 2:  # give_hard_exercise
            success_prob = max(0.05, mastery_now - 0.2)
            if self.np_random.random() < success_prob:
                mastery_delta    = lr * 1.2
                engagement_delta = 0.08
                got_correct      = True
            else:
                mastery_delta    = 0.0
                engagement_delta = -0.06
                got_wrong        = True

        elif action == 3:  # provide_hint
            mastery_delta    = lr * 0.3
            engagement_delta = 0.01

        elif action == 4:  # give_feedback
            mastery_delta    = lr * 0.5 * (1 - mastery_now)
            engagement_delta = 0.03

        elif action == 5:  # reduce_difficulty
            self.difficulty  = max(0.1, self.difficulty - 0.15)
            engagement_delta = 0.05

        elif action == 6:  # advance_topic
            if mastery_now >= 0.75:
                engagement_delta = 0.06
            else:
                engagement_delta = -0.05

        elif action == 7:  # run_assessment
            mastery_delta    = lr * 0.2
            engagement_delta = -0.01 if mastery_now < 0.5 else 0.02

        return mastery_delta, engagement_delta, got_correct, got_wrong

    def step(self, action):
        if self.done:
            raise RuntimeError("Called step() on a finished episode. Call reset() first.")

        self.steps      += 1
        self.time_on_topic += 1
        self.last_action = action
        reward           = 0.0
        truncated        = False

        prev_mastery    = self.mastery[self.current_topic]
        prev_engagement = self.engagement

        mastery_delta, engage_delta, got_correct, got_wrong = \
            self._simulate_student_response(action)

        # Apply updates with clipping
        self.mastery[self.current_topic] = float(np.clip(
            self.mastery[self.current_topic] + mastery_delta, 0.0, 1.0
        ))
        self.engagement = float(np.clip(self.engagement + engage_delta, 0.0, 1.0))

        # Update streaks
        if got_correct:
            self.consec_correct += 1
            self.consec_wrong    = 0
            self.error_rate      = max(0.0, self.error_rate - 0.08)
        elif got_wrong:
            self.consec_wrong   += 1
            self.consec_correct  = 0
            self.error_rate      = min(1.0, self.error_rate + 0.10)

        # Penalise giving a hard exercise when student is struggling
        if action == 2 and self.mastery[self.current_topic] < 0.3:
            reward -= 1.0

        # Reward mastery improvement
        mastery_gain = self.mastery[self.current_topic] - prev_mastery
        if mastery_gain > 0:
            reward += mastery_gain * 2.0

        # Reward engagement improvement
        engage_gain = self.engagement - prev_engagement
        if engage_gain > 0:
            reward += engage_gain * 0.5

        # Frustration penalty
        if self.consec_wrong >= self.profile["frustration_thresh"]:
            reward -= 1.5

        # Topic advancement
        if action == 6 and self.mastery[self.current_topic] >= 0.75:
            reward += 5.0
            self.topics_completed += 1
            if self.current_topic < NUM_TOPICS - 1:
                self.current_topic  += 1
                self.time_on_topic   = 0
                self.consec_correct  = 0
                self.consec_wrong    = 0
                self.difficulty      = 0.3
            else:
                reward    += 10.0
                self.done  = True

        # Advancing too early without mastery
        elif action == 6 and self.mastery[self.current_topic] < 0.75:
            reward -= 2.0

        # Small time penalty to discourage spinning in place
        reward -= 0.3

        # Engagement decay each step (simulates natural fatigue)
        decay = self.profile["engage_decay"]
        self.engagement = float(np.clip(self.engagement - decay, 0.0, 1.0))

        # Dropout condition
        if self.engagement < 0.1 or (self.consec_wrong >= 6):
            reward   -= 8.0
            self.done = True

        # Max steps timeout
        if self.steps >= self.max_steps:
            truncated = True
            self.done = True

        self.last_reward = reward
        self.history.append({
            "step":        self.steps,
            "action":      ACTIONS[action],
            "mastery":     round(float(self.mastery[self.current_topic]), 3),
            "engagement":  round(self.engagement, 3),
            "reward":      round(reward, 3),
            "topic":       TOPICS[self.current_topic],
        })

        obs  = self._get_obs()
        info = {
            "topic":             TOPICS[self.current_topic],
            "mastery":           float(self.mastery[self.current_topic]),
            "engagement":        self.engagement,
            "topics_completed":  self.topics_completed,
            "action_name":       ACTIONS[action],
        }

        return obs, reward, self.done, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from environment.rendering import EduCodeRenderer
                self.renderer = EduCodeRenderer(
                    num_topics=NUM_TOPICS,
                    topic_names=TOPICS,
                    action_names=list(ACTIONS.values()),
                )
            self.renderer.draw(
                topic_idx   = self.current_topic,
                mastery     = self.mastery,
                engagement  = self.engagement,
                error_rate  = self.error_rate,
                action      = self.last_action,
                reward      = self.last_reward,
                step        = self.steps,
                topics_done = self.topics_completed,
                difficulty  = self.difficulty,
            )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

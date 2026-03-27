import pygame
import sys
import math


WIDTH  = 960
HEIGHT = 620

# Colour palette
BG_COLOUR        = (15,  20,  35)
PANEL_COLOUR     = (25,  32,  55)
ACCENT_BLUE      = (52, 152, 219)
ACCENT_GREEN     = (39, 174,  96)
ACCENT_RED       = (231, 76,  60)
ACCENT_ORANGE    = (230, 126,  34)
ACCENT_PURPLE    = (155,  89, 182)
TEXT_WHITE       = (236, 240, 241)
TEXT_GREY        = (127, 140, 141)
BAR_BG           = (44,  62,  80)
HIGHLIGHT        = (241, 196,  15)


def _bar(surface, x, y, w, h, value, colour, bg=BAR_BG):
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=4)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        pygame.draw.rect(surface, colour, (x, y, fill, h), border_radius=4)


def _label(surface, font, text, x, y, colour=TEXT_WHITE, align="left"):
    surf = font.render(text, True, colour)
    if align == "right":
        x = x - surf.get_width()
    elif align == "center":
        x = x - surf.get_width() // 2
    surface.blit(surf, (x, y))
    return surf.get_width()


class EduCodeRenderer:
    """
    Pygame-based visualiser for the EduCode Rwanda tutor environment.

    The window is divided into three panels:
      Left   - student profile, current topic, engagement and error bars
      Centre - topic mastery progress for all 8 JavaScript topics
      Right  - last action taken, reward received, step counter
    """

    def __init__(self, num_topics, topic_names, action_names):
        pygame.init()
        pygame.display.set_caption("EduCode Rwanda - AI Tutor Simulation")

        self.num_topics   = num_topics
        self.topic_names  = topic_names
        self.action_names = action_names
        self.screen       = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock        = pygame.time.Clock()

        self.font_title  = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_body   = pygame.font.SysFont("monospace", 13)
        self.font_small  = pygame.font.SysFont("monospace", 11)
        self.font_large  = pygame.font.SysFont("monospace", 28, bold=True)

        self.reward_history    = []
        self.max_history       = 120

    def draw(self, topic_idx, mastery, engagement, error_rate,
             action, reward, step, topics_done, difficulty):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(BG_COLOUR)

        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

        self._draw_header(step, topics_done)
        self._draw_left_panel(topic_idx, engagement, error_rate, difficulty)
        self._draw_centre_panel(mastery, topic_idx)
        self._draw_right_panel(action, reward)
        self._draw_reward_graph()

        pygame.display.flip()
        self.clock.tick(30)

    def _draw_header(self, step, topics_done):
        pygame.draw.rect(self.screen, PANEL_COLOUR, (0, 0, WIDTH, 52))
        _label(self.screen, self.font_title,
               "EduCode Rwanda  |  AI Tutor Agent",
               20, 10, ACCENT_BLUE)
        _label(self.screen, self.font_body,
               f"Step: {step:>4}     Topics completed: {topics_done} / 8",
               WIDTH - 20, 16, TEXT_GREY, align="right")
        pygame.draw.line(self.screen, ACCENT_BLUE, (0, 52), (WIDTH, 52), 1)

    def _draw_left_panel(self, topic_idx, engagement, error_rate, difficulty):
        px, py, pw = 18, 70, 260

        pygame.draw.rect(self.screen, PANEL_COLOUR,
                         (px - 8, py - 8, pw + 16, 310), border_radius=8)

        _label(self.screen, self.font_title, "Student State", px, py, ACCENT_GREEN)
        py += 30

        current = self.topic_names[topic_idx].replace("_", " ").title()
        _label(self.screen, self.font_body, "Current Topic", px, py, TEXT_GREY)
        py += 18
        _label(self.screen, self.font_body, current, px, py, TEXT_WHITE)
        py += 28

        # Engagement bar
        _label(self.screen, self.font_body, f"Engagement  {engagement:.0%}", px, py, TEXT_GREY)
        py += 16
        eng_colour = ACCENT_GREEN if engagement > 0.5 else ACCENT_ORANGE if engagement > 0.25 else ACCENT_RED
        _bar(self.screen, px, py, pw, 14, engagement, eng_colour)
        py += 28

        # Error rate bar
        _label(self.screen, self.font_body, f"Error Rate  {error_rate:.0%}", px, py, TEXT_GREY)
        py += 16
        err_colour = ACCENT_RED if error_rate > 0.6 else ACCENT_ORANGE if error_rate > 0.3 else ACCENT_GREEN
        _bar(self.screen, px, py, pw, 14, error_rate, err_colour)
        py += 28

        # Difficulty bar
        _label(self.screen, self.font_body, f"Difficulty  {difficulty:.0%}", px, py, TEXT_GREY)
        py += 16
        _bar(self.screen, px, py, pw, 14, difficulty, ACCENT_PURPLE)
        py += 36

        # Student status indicator
        if engagement > 0.6:
            status, s_colour = "Engaged", ACCENT_GREEN
        elif engagement > 0.3:
            status, s_colour = "Neutral", ACCENT_ORANGE
        else:
            status, s_colour = "At Risk", ACCENT_RED

        pygame.draw.circle(self.screen, s_colour, (px + 8, py + 8), 7)
        _label(self.screen, self.font_body, f"  Status: {status}", px, py, s_colour)

    def _draw_centre_panel(self, mastery, topic_idx):
        cx, cy, cw = 310, 70, 360

        pygame.draw.rect(self.screen, PANEL_COLOUR,
                         (cx - 8, cy - 8, cw + 16, 390), border_radius=8)

        _label(self.screen, self.font_title, "Topic Mastery Progress", cx, cy, ACCENT_BLUE)
        cy += 34

        bar_h   = 28
        spacing = 10

        for i, name in enumerate(self.topic_names):
            m = float(mastery[i])
            label_text = name.replace("_", " ").title()

            if i == topic_idx:
                pygame.draw.rect(self.screen, (35, 45, 70),
                                 (cx - 6, cy - 3, cw + 4, bar_h + 6), border_radius=6)

            _label(self.screen, self.font_small, label_text, cx, cy + 7, TEXT_GREY)

            bar_x = cx + 170
            bar_w = 160

            if m >= 0.75:
                colour = ACCENT_GREEN
            elif m >= 0.4:
                colour = ACCENT_ORANGE
            else:
                colour = ACCENT_RED

            _bar(self.screen, bar_x, cy + 6, bar_w, 16, m, colour)
            _label(self.screen, self.font_small, f"{m:.0%}",
                   bar_x + bar_w + 6, cy + 7, TEXT_WHITE)

            if i == topic_idx:
                pygame.draw.polygon(self.screen, HIGHLIGHT, [
                    (cx - 14, cy + bar_h // 2),
                    (cx - 6,  cy + bar_h // 2 - 5),
                    (cx - 6,  cy + bar_h // 2 + 5),
                ])

            cy += bar_h + spacing

    def _draw_right_panel(self, action, reward):
        rx, ry, rw = 700, 70, 242

        pygame.draw.rect(self.screen, PANEL_COLOUR,
                         (rx - 8, ry - 8, rw + 16, 310), border_radius=8)

        _label(self.screen, self.font_title, "Agent Action", rx, ry, ACCENT_ORANGE)
        ry += 30

        if action is not None:
            action_text = self.action_names[action].replace("_", " ").upper()
        else:
            action_text = "WAITING"

        words = action_text.split()
        line1 = " ".join(words[:2]) if len(words) >= 2 else action_text
        line2 = " ".join(words[2:]) if len(words) > 2 else ""

        _label(self.screen, self.font_large, line1, rx, ry, ACCENT_ORANGE)
        if line2:
            ry += 34
            _label(self.screen, self.font_large, line2, rx, ry, ACCENT_ORANGE)
        ry += 50

        pygame.draw.line(self.screen, BAR_BG, (rx, ry), (rx + rw, ry), 1)
        ry += 16

        _label(self.screen, self.font_title, "Reward", rx, ry, TEXT_GREY)
        ry += 28

        r_colour = ACCENT_GREEN if reward > 0 else ACCENT_RED
        _label(self.screen, self.font_large, f"{reward:+.2f}", rx, ry, r_colour)
        ry += 52

        pygame.draw.line(self.screen, BAR_BG, (rx, ry), (rx + rw, ry), 1)
        ry += 16

        _label(self.screen, self.font_body, "Action Legend", rx, ry, TEXT_GREY)
        ry += 20
        for i, name in enumerate(self.action_names):
            _label(self.screen, self.font_small,
                   f"{i}  {name.replace('_', ' ')}", rx, ry, TEXT_GREY)
            ry += 16

    def _draw_reward_graph(self):
        gx, gy   = 18, 410
        gw, gh   = 924, 140
        padding  = 12

        pygame.draw.rect(self.screen, PANEL_COLOUR,
                         (gx - 8, gy - 8, gw + 16, gh + 24), border_radius=8)
        _label(self.screen, self.font_title, "Reward History", gx, gy, TEXT_GREY)
        gy += 24

        if len(self.reward_history) < 2:
            return

        max_r = max(abs(r) for r in self.reward_history) or 1.0
        zero_y = gy + gh // 2

        pygame.draw.line(self.screen, BAR_BG, (gx, zero_y), (gx + gw, zero_y), 1)

        pts = []
        for i, r in enumerate(self.reward_history):
            x = gx + int(i / (self.max_history - 1) * gw)
            y = zero_y - int((r / max_r) * (gh // 2 - padding))
            pts.append((x, y))

        for i in range(len(pts) - 1):
            colour = ACCENT_GREEN if self.reward_history[i] >= 0 else ACCENT_RED
            pygame.draw.line(self.screen, colour, pts[i], pts[i + 1], 2)

        _label(self.screen, self.font_small, f"+{max_r:.1f}",
               gx, gy, TEXT_GREY)
        _label(self.screen, self.font_small, f"-{max_r:.1f}",
               gx, gy + gh - padding, TEXT_GREY)

    def close(self):
        pygame.quit()

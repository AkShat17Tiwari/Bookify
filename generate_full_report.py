#!/usr/bin/env python3
"""
BOOKIFY - Concise Project Report Generator (~20 pages)
Generates a professional PDF with university cover pages, charts, and screenshots.
"""

import pickle, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import Counter
from fpdf import FPDF
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────
OUT_DIR = 'report_assets'
os.makedirs(OUT_DIR, exist_ok=True)

STUDENT_NAME = 'Akshat Tiwari'
STUDENT_REG = '23FE10CSE00156'
GUIDE_NAME = 'Dr. Susheela Vishnoi'
PROJECT_TITLE = 'Bookify'
SEMESTER = 'VI'
YEAR = '2025-2026'
PERIOD = 'Jan-May 2026'
LOGO_PATH = f'{OUT_DIR}/manipal_logo.png'

# Colors
C_BG     = '#0a0a0f'
C_GREEN  = '#00e68a'
C_CYAN   = '#00b4d8'
C_PURPLE = '#a78bfa'
C_AMBER  = '#f59e0b'
C_RED    = '#f87171'
C_WHITE  = '#f0f0f0'
C_GRAY   = '#888888'
PALETTE  = ['#00e68a','#00b4d8','#a78bfa','#f59e0b','#f87171',
            '#38bdf8','#4ade80','#fb923c','#c084fc','#f472b6']

plt.rcParams.update({
    'figure.facecolor': C_BG, 'axes.facecolor': '#111118',
    'axes.edgecolor': '#333', 'text.color': C_WHITE,
    'axes.labelcolor': C_WHITE, 'xtick.color': C_GRAY,
    'ytick.color': C_GRAY, 'grid.color': '#222', 'grid.alpha': 0.4,
    'font.family': 'sans-serif', 'font.size': 11,
})

# ─── Load Data ────────────────────────────────────────────────────
print("Loading project data...")
with open('popular.pkl','rb') as f: pop = pickle.load(f)
with open('pt.pkl','rb') as f: pt = pickle.load(f)
with open('books_slim.pkl','rb') as f: books_slim = pickle.load(f)
with open('model_accuracy.json','r') as f: acc = json.load(f)
with open('genre_data.pkl','rb') as f: genres = pickle.load(f)
with open('ncf_similarity_scores.pkl','rb') as f: ncf = pickle.load(f)

if isinstance(genres, dict) and 'genre_map' in genres:
    genre_map = genres['genre_map']
elif isinstance(genres, dict):
    genre_map = genres
else:
    genre_map = {}

genre_counts = Counter()
for title, glist in genre_map.items():
    for g in glist:
        genre_counts[g] += 1

classic_acc_data = {}
if os.path.exists('classic_accuracy.json'):
    with open('classic_accuracy.json','r') as f: classic_acc_data = json.load(f)

# ═══════════════════════════════════════════════════════════════════
# CHART GENERATION (Compact)
# ═══════════════════════════════════════════════════════════════════

# Chart 1: Genre Distribution
print("  Chart 1: Genre distribution...")
fig, ax = plt.subplots(figsize=(10, 7))
gs = genre_counts.most_common()
names = [g[0] for g in gs][::-1]; counts = [g[1] for g in gs][::-1]
colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))][::-1]
bars = ax.barh(names, counts, color=colors, height=0.7, edgecolor='none')
ax.set_xlabel('Number of Books', fontsize=12, fontweight='bold')
ax.set_title('Genre Distribution Across 4,659 Books', fontsize=16, fontweight='bold', color=C_GREEN, pad=15)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width()+15, bar.get_y()+bar.get_height()/2, str(count), va='center', fontsize=9, color=C_GRAY)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/genre_distribution.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 2: Top 15 Books
print("  Chart 2: Top books...")
fig, ax = plt.subplots(figsize=(10, 6))
top15 = pop.head(15).sort_values('num_ratings')
labels = [t[:30]+'...' if len(t)>30 else t for t in top15['Book-Title']]
votes = top15['num_ratings'].values; ratings = top15['avg_rating'].values
nr = (ratings - ratings.min()) / (ratings.max() - ratings.min())
bars = ax.barh(labels, votes, color=[plt.cm.YlOrRd(0.3+0.6*n) for n in nr], height=0.65, edgecolor='none')
ax.set_xlabel('Number of Ratings', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Rated Books', fontsize=16, fontweight='bold', color=C_AMBER, pad=15)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/top_books_votes.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 3: Rating Distribution
print("  Chart 3: Rating distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
ar = pop['avg_rating'].values / 2
ax.hist(ar, bins=12, color=C_CYAN, edgecolor='#111118', alpha=0.85, linewidth=1.5)
ax.axvline(np.mean(ar), color=C_GREEN, linestyle='--', linewidth=2, label=f'Mean: {np.mean(ar):.2f}')
ax.set_xlabel('Rating (out of 5)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Books', fontsize=12, fontweight='bold')
ax.set_title('Rating Distribution - Top 50 Popular Books', fontsize=14, fontweight='bold', color=C_CYAN, pad=15)
ax.legend(fontsize=11, facecolor='#111118', edgecolor='#333')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/rating_distribution.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 4: Training Loss
print("  Chart 4: Training loss...")
fig, ax = plt.subplots(figsize=(8, 5))
epochs = list(range(1, acc['training']['epochs_run']+1))
tl = acc['training']['train_losses']; vl = acc['training']['val_losses']
ax.plot(epochs, tl, color=C_GREEN, linewidth=2.5, marker='o', markersize=5, label='Training Loss', zorder=5)
ax.plot(epochs, vl, color=C_PURPLE, linewidth=2.5, marker='s', markersize=5, label='Validation Loss', zorder=5)
ax.fill_between(epochs, tl, alpha=0.1, color=C_GREEN)
ax.fill_between(epochs, vl, alpha=0.1, color=C_PURPLE)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
ax.set_title('NCF Model Training Progress', fontsize=14, fontweight='bold', color=C_GREEN, pad=15)
ax.legend(fontsize=10, facecolor='#111118', edgecolor='#333')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/training_loss.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 5: Performance Metrics Gauges
print("  Chart 5: Performance metrics...")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
metrics = [('Accuracy', acc['accuracy']['accuracy_pct'], '%', C_GREEN),
           ('RMSE', acc['accuracy']['rmse'], '', C_CYAN),
           ('MAE', acc['accuracy']['mae'], '', C_PURPLE),
           ('NRMSE', acc['accuracy']['nrmse'], '', C_AMBER)]
for ax, (name, value, unit, color) in zip(axes, metrics):
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2); ax.set_aspect('equal'); ax.axis('off')
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='#333', linewidth=8, solid_capstyle='round')
    if name=='Accuracy': frac=value/100
    elif name in ('RMSE','MAE'): frac=max(0,1-value/5)
    else: frac=max(0,1-value)
    tf = np.linspace(np.pi/2, np.pi/2-2*np.pi*frac, 100)
    ax.plot(np.cos(tf), np.sin(tf), color=color, linewidth=8, solid_capstyle='round')
    ax.text(0, 0.05, f'{value}{unit}', ha='center', va='center', fontsize=18, fontweight='bold', color=color)
    ax.text(0, -0.35, name, ha='center', va='center', fontsize=11, fontweight='bold', color=C_GRAY)
plt.suptitle('NCF Model Performance Metrics', fontsize=14, fontweight='bold', color=C_WHITE, y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/performance_metrics.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 6: Dataset Split
print("  Chart 6: Dataset split...")
fig, ax = plt.subplots(figsize=(6, 6))
sizes = [acc['dataset']['train_size'], acc['dataset']['val_size'], acc['dataset']['test_size']]
ls = [f"Train\n{sizes[0]:,}", f"Validation\n{sizes[1]:,}", f"Test\n{sizes[2]:,}"]
wedges, texts, autotexts = ax.pie(sizes, labels=ls, colors=[C_GREEN,C_CYAN,C_PURPLE],
    autopct='%1.1f%%', startangle=90, textprops={'fontsize':12,'color':C_WHITE},
    wedgeprops={'edgecolor':C_BG,'linewidth':2})
for at in autotexts: at.set_fontweight('bold'); at.set_fontsize(11)
ax.set_title('Training / Validation / Test Split', fontsize=14, fontweight='bold', color=C_CYAN, pad=15)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/dataset_split.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 7: Similarity Heatmap
print("  Chart 7: Similarity heatmap...")
fig, ax = plt.subplots(figsize=(8, 7))
np.random.seed(42); si = np.random.choice(ncf.shape[0], 30, replace=False); si.sort()
im = ax.imshow(ncf[np.ix_(si, si)], cmap='inferno', aspect='auto', interpolation='nearest')
ax.set_title('NCF Similarity Score Heatmap (30-Book Sample)', fontsize=14, fontweight='bold', color=C_AMBER, pad=15)
ax.set_xlabel('Book Index', fontsize=11); ax.set_ylabel('Book Index', fontsize=11)
cbar = plt.colorbar(im, ax=ax, shrink=0.85); cbar.set_label('Cosine Similarity', fontsize=11, color=C_WHITE)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/similarity_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 8: Architecture Diagram
print("  Chart 8: Architecture...")
fig, ax = plt.subplots(figsize=(10, 6)); ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis('off')
ax.set_title('Neural Collaborative Filtering - Architecture', fontsize=16, fontweight='bold', color=C_GREEN, pad=20)
def draw_box(ax, x, y, w, h, label, color, sub=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",facecolor=color,edgecolor='white',linewidth=1.2,alpha=0.25))
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",facecolor='none',edgecolor=color,linewidth=1.8))
    ax.text(x+w/2, y+h/2+(0.12 if sub else 0), label, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
    if sub: ax.text(x+w/2, y+h/2-0.2, sub, ha='center', va='center', fontsize=8, color=C_GRAY)
def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.5))
draw_box(ax,.3,4.5,1.8,1.2,'User ID',C_CYAN,'Input'); draw_box(ax,.3,2.3,1.8,1.2,'Book ID',C_AMBER,'Input')
draw_box(ax,3,4.5,1.8,1.2,'User Embed',C_CYAN,'dim=64'); draw_box(ax,3,2.3,1.8,1.2,'Book Embed',C_AMBER,'dim=64')
draw_box(ax,5.5,3.2,1.5,1.5,'Concat',C_PURPLE,'dim=128')
draw_box(ax,7.5,4.8,1.5,.8,'FC: 128',C_GREEN,'ReLU+BN+Drop'); draw_box(ax,7.5,3.6,1.5,.8,'FC: 64',C_GREEN,'ReLU+BN+Drop')
draw_box(ax,7.5,2.4,1.5,.8,'FC: 32',C_GREEN,'ReLU+BN+Drop'); draw_box(ax,7.5,1.,1.5,.8,'Output',C_RED,'Sigmoid')
for a in [(2.1,5.1,3,5.1),(2.1,2.9,3,2.9),(4.8,5.1,5.5,4.2),(4.8,2.9,5.5,3.7),(7,3.95,7.5,5.2),(7,3.95,7.5,4),(7,3.95,7.5,2.8),(8.25,4.8,8.25,4.4),(8.25,3.6,8.25,3.2),(8.25,2.4,8.25,1.8)]:
    draw_arrow(ax, *a)
ax.text(5, 0.6, f'Total Parameters: {acc["total_parameters"]:,}', ha='center', fontsize=11, color=C_GRAY, style='italic')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/architecture.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 9: Sparsity
print("  Chart 9: Sparsity...")
fig, ax = plt.subplots(figsize=(8, 5))
tc = pt.shape[0]*pt.shape[1]; nz = np.count_nonzero(pt.values); z = tc-nz; sp = z/tc*100
ax.barh(['Interactions\n(Non-Zero)', 'Empty Cells\n(Zero)'], [nz, z], color=[C_GREEN,'#222230'], edgecolor='none', height=0.5)
ax.set_xscale('log'); ax.set_xlabel('Count (log scale)', fontsize=12, fontweight='bold')
ax.set_title(f'User-Book Matrix Sparsity: {sp:.2f}%', fontsize=14, fontweight='bold', color=C_RED, pad=15)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(nz*1.5, 0, f'{nz:,}', va='center', fontsize=12, fontweight='bold', color=C_GREEN)
ax.text(z*1.5, 1, f'{z:,}', va='center', fontsize=12, fontweight='bold', color=C_GRAY)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/sparsity.png', dpi=150, bbox_inches='tight'); plt.close()

# Chart 10: Classic vs NCF
print("  Chart 10: Classic vs NCF...")
fig, ax = plt.subplots(figsize=(9, 5))
cats = ['Precision@10', 'NDCG@10', 'Hit Rate@10']
cv = [classic_acc_data.get('precision_at_10',0.193), classic_acc_data.get('ndcg_at_10',0.401), classic_acc_data.get('hit_rate_at_10',0.781)]
nv = [acc['ranking']['precision_at_10'], acc['ranking']['ndcg_at_10'], acc['ranking']['hit_rate_at_10']]
x = np.arange(len(cats)); w = 0.35
b1 = ax.bar(x-w/2, cv, w, label='Classic (Cosine)', color=C_CYAN, alpha=0.85)
b2 = ax.bar(x+w/2, nv, w, label='NCF (Deep Learning)', color=C_GREEN, alpha=0.85)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Classic vs NCF Ranking Metrics', fontsize=14, fontweight='bold', color=C_CYAN, pad=15)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.legend(fontsize=10, facecolor='#111118', edgecolor='#333')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/classic_vs_ncf.png', dpi=150, bbox_inches='tight'); plt.close()

print("  All charts generated!")

# ═══════════════════════════════════════════════════════════════════
# PDF GENERATION - CONCISE ~20 PAGES
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating PDF report...")

class PDF(FPDF):
    def header(self):
        if self.page_no() > 4:
            self.set_font('Times', 'B', 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, 'BOOKIFY - Intelligent Book Recommendation System', align='L')
            self.cell(0, 8, f'Page {self.page_no() - 4}', align='R', new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(139, 0, 0)
            self.set_line_width(0.3)
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self): pass

    def add_logo_with_name(self):
        """Add Manipal logo + university name centered."""
        logo_w = 35
        text = 'MANIPAL UNIVERSITY JAIPUR'
        self.set_font('Times', 'B', 14)
        text_w = self.get_string_width(text)
        total_w = logo_w + 4 + text_w
        start_x = (210 - total_w) / 2
        y = self.get_y()
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, x=start_x, y=y, w=logo_w)
        self.set_xy(start_x + logo_w + 4, y + 2)
        self.set_font('Times', 'B', 16)
        self.set_text_color(139, 0, 0)
        self.cell(text_w + 10, 8, text)
        self.ln(16)

    def section_title(self, num, title):
        self.set_font('Times', 'B', 16)
        self.set_text_color(139, 0, 0)
        self.cell(0, 10, f'{num}. {title}', new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(139, 0, 0)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font('Times', 'B', 12)
        self.set_text_color(0, 51, 102)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font('Times', '', 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=12):
        self.set_font('Times', '', 10.5)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, '-')
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def kv_row(self, key, value, bold_val=False):
        self.set_font('Times', '', 10.5)
        self.set_text_color(80, 80, 80)
        self.cell(60, 6, key)
        self.set_font('Times', 'B' if bold_val else '', 10.5)
        self.set_text_color(20, 20, 20)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def add_chart(self, path, w=180):
        if os.path.exists(path):
            x = (210 - w) / 2
            self.image(path, x=x, w=w)
            self.ln(4)


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=18)

# ═══════════════════════════════════════════════════════════════════
# PAGE 1: TITLE PAGE
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(12)
pdf.set_font('Times', 'I', 14); pdf.set_text_color(0,0,0)
pdf.cell(0, 8, 'A  Report', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', '', 14)
pdf.cell(0, 8, 'on', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Times', 'B', 28)
pdf.cell(0, 14, 'Bookify', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Times', 'I', 12)
pdf.cell(0, 7, 'carried out as part of the course Project Based Learning- 4', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, 'Submitted by', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Times', 'BI', 15)
pdf.cell(0, 8, STUDENT_NAME, align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.set_font('Times', 'BI', 14)
pdf.cell(0, 8, STUDENT_REG, align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.cell(0, 8, f'{SEMESTER} Semester', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font('Times', 'I', 12)
pdf.cell(0, 7, 'in partial fulfilment for the award of the degree', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', 'I', 13)
pdf.cell(0, 7, 'of', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Times', 'B', 16)
pdf.cell(0, 10, 'BACHELOR OF TECHNOLOGY', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.set_font('Times', '', 12)
pdf.cell(0, 7, 'In', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.set_font('Times', 'B', 14)
pdf.cell(0, 8, 'Computer Science & Engineering', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(16)
# Logo + University Name
pdf.add_logo_with_name()
pdf.set_font('Times', 'I', 9)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 5, '(University under Section 2(f) of the UGC Act)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(12)
pdf.set_font('Times', 'B', 11); pdf.set_text_color(0,0,0)
pdf.cell(0, 7, 'Department of Computer Science & Engineering,', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, 'School of Computer Science and Engineering,', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', 'B', 13)
pdf.cell(0, 8, 'Manipal University Jaipur,', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', 'BI', 12); pdf.set_text_color(139,0,0)
pdf.cell(0, 8, PERIOD, align='C', new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════════════════
# PAGE 2: GUIDE PAGE
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(12)
pdf.set_font('Times', 'I', 14); pdf.set_text_color(0,0,0)
pdf.cell(0, 8, 'A  Report', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', '', 14)
pdf.cell(0, 8, 'on', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Times', 'B', 28)
pdf.cell(0, 14, 'Bookify', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Times', 'I', 12)
pdf.cell(0, 7, 'carried out as part of the course Project Based Learning-2/4', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, 'Submitted by', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Times', 'BI', 15)
pdf.cell(0, 8, STUDENT_NAME, align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.set_font('Times', 'BI', 14)
pdf.cell(0, 8, STUDENT_REG, align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.cell(0, 8, f'{SEMESTER} Semester', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font('Times', 'I', 12)
pdf.cell(0, 7, 'in partial fulfilment for the award of the degree', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', 'I', 13)
pdf.cell(0, 7, 'of', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Times', 'B', 16)
pdf.cell(0, 10, 'BACHELOR OF TECHNOLOGY', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.set_font('Times', '', 12)
pdf.cell(0, 7, 'In', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.set_font('Times', 'B', 14)
pdf.cell(0, 8, 'Computer Science & Engineering', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(18)
pdf.set_font('Times', 'I', 13); pdf.set_text_color(0,0,0)
pdf.cell(0, 8, 'Under the Guidance of :', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Times', '', 13)
pdf.cell(0, 8, f'Guide Name :  {GUIDE_NAME}', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.cell(0, 8, 'Guide Signature(with date) : ............................................', new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════════════════
# PAGE 3: ACKNOWLEDGEMENT
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(6)
pdf.set_font('Times', 'B', 20); pdf.set_text_color(0,0,0)
pdf.cell(0, 12, 'Acknowledgement', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.set_font('Times', '', 12); pdf.set_text_color(30,30,30)
pdf.multi_cell(0, 7,
    'This project would not have been completed without the help, support, comments, advice, '
    'cooperation and coordination of various people. However, it is impossible to thank everyone '
    'individually; I am hereby making a humble effort to thank some of them.')
pdf.ln(5)
pdf.multi_cell(0, 7,
    f'I acknowledge and express my deepest sense of gratitude to my internal supervisor {GUIDE_NAME} '
    'for her constant support, guidance, and continuous engagement. I highly appreciate her technical '
    f'comments, suggestions, and criticism during the progress of this project titled "Bookify".')
pdf.ln(5)
pdf.set_font('Times', '', 12)
pdf.write(7, 'I owe my profound gratitude to ')
pdf.set_font('Times', 'BI', 12)
pdf.write(7, 'Dr. Neha Chaudhary')
pdf.set_font('Times', '', 12)
pdf.write(7, ' , Head, Department of CSE, for her valuable guidance and for facilitating me during my work. ')
pdf.write(7, 'I am also very grateful to all the faculty members and staff for their precious support and cooperation during the development of this project.')
pdf.ln(8)
pdf.multi_cell(0, 7, 'Finally, I extend my heartfelt appreciation to my classmates for their help and encouragement.')
pdf.ln(6)
pdf.set_font('Times', 'B', 14); pdf.set_text_color(0,0,0)
pdf.cell(0, 8, STUDENT_NAME, align='R', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', 'B', 13)
pdf.cell(0, 8, STUDENT_REG, align='R', new_x="LMARGIN", new_y="NEXT")
pdf.ln(18)
# Logo + University
pdf.add_logo_with_name()
pdf.set_font('Times', 'I', 9); pdf.set_text_color(100,100,100)
pdf.cell(0, 5, '(University under Section 2(f) of the UGC Act)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font('Times', 'B', 12); pdf.set_text_color(0,0,0)
pdf.cell(0, 7, 'Department of Computer Science and Engineering', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 7, 'School of Computer Science and Engineering', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Times', '', 12)
pdf.cell(0, 7, 'Date: ______________', align='R', new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════════════════
# PAGE 4: CERTIFICATE
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(10)
pdf.set_font('Times', 'BU', 22); pdf.set_text_color(0,0,0)
pdf.cell(0, 14, 'CERTIFICATE', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(12)
pdf.set_font('Times', '', 12); pdf.set_text_color(30,30,30)
pdf.write(7, 'This is to certify that the project entitled "')
pdf.set_font('Times', 'BI', 12); pdf.write(7, 'Bookify')
pdf.set_font('Times', '', 12)
pdf.write(7, '" is a bonafide work carried out as PBL-4 ')
pdf.set_font('Times', 'BI', 12); pdf.write(7, 'End Term Assessment')
pdf.set_font('Times', '', 12)
pdf.write(7, ' in partial fulfillment for the award of the degree of Bachelor of Technology in Computer Science and Engineering, by ')
pdf.set_font('Times', 'BIU', 12); pdf.write(7, f'{STUDENT_NAME} ')
pdf.set_font('Times', '', 12)
pdf.write(7, f'bearing registration number {STUDENT_REG}, during the academic semester {SEMESTER} ')
pdf.set_font('Times', 'I', 12); pdf.write(7, f'of year {YEAR}.')
pdf.ln(25)
pdf.set_font('Times', '', 12); pdf.set_text_color(0,0,0)
pdf.cell(0, 8, 'Place: Manipal University Jaipur, Jaipur', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.cell(0, 8, f'Name of the project guide: {GUIDE_NAME}', new_x="LMARGIN", new_y="NEXT")
pdf.ln(50)
pdf.cell(0, 8, 'Signature of the project guide: _______________________', new_x="LMARGIN", new_y="NEXT")

# ═══════════════════════════════════════════════════════════════════
# PAGE 5: TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(2)
pdf.set_font('Times', 'B', 20); pdf.set_text_color(139,0,0)
pdf.cell(0, 12, 'Table of Contents', new_x="LMARGIN", new_y="NEXT")
pdf.set_draw_color(139,0,0); pdf.set_line_width(0.4); pdf.line(10, pdf.get_y(), 75, pdf.get_y())
pdf.ln(6)
toc = [('1','Abstract'),('2','Introduction'),('3','Dataset & Data Analysis'),
       ('4','Methodology - NCF Model'),('5','Results & Performance Evaluation'),
       ('6','System Architecture & Features'),('7','Technology Stack'),
       ('8','Application Screenshots'),('9','Conclusion & Future Work'),('10','References')]
for num, title in toc:
    pdf.set_font('Times', '', 12); pdf.set_text_color(40,40,40)
    pdf.cell(12, 9, num + '.'); pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")

pdf.ln(8)
pdf.set_font('Times', 'B', 16); pdf.set_text_color(139,0,0)
pdf.cell(0, 10, 'List of Figures', new_x="LMARGIN", new_y="NEXT")
pdf.set_draw_color(139,0,0); pdf.set_line_width(0.3); pdf.line(10, pdf.get_y(), 55, pdf.get_y())
pdf.ln(4)
figs = [('Fig 1.','Genre Distribution'),('Fig 2.','Top 15 Most Rated Books'),
        ('Fig 3.','Rating Distribution'),('Fig 4.','NCF Training Loss Curve'),
        ('Fig 5.','Model Performance Gauges'),('Fig 6.','Dataset Split'),
        ('Fig 7.','Similarity Heatmap'),('Fig 8.','NCF Architecture'),
        ('Fig 9.','Matrix Sparsity'),('Fig 10.','Classic vs NCF Comparison'),
        ('Fig 11-16.','Application Screenshots')]
for num, title in figs:
    pdf.set_font('Times', '', 11); pdf.set_text_color(60,60,60)
    pdf.cell(16, 6, num); pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: ABSTRACT
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('1', 'Abstract')
pdf.body_text(
    'BOOKIFY is an intelligent book recommendation system leveraging Neural Collaborative '
    'Filtering (NCF) for personalized suggestions. Built on the Book-Crossing dataset with '
    f'{acc["dataset"]["total_ratings"]:,} ratings across {acc["dataset"]["total_books"]:,} books '
    f'and {acc["dataset"]["total_users"]:,} users, the NCF model achieves {acc["accuracy"]["accuracy_pct"]}% '
    f'accuracy with RMSE {acc["accuracy"]["rmse"]}. The system offers dual recommendation modes, '
    'genre exploration (23 categories), mood-based recommendations via facial emotion detection, '
    'multi-modal search, and explainable AI. Built with Flask, React/TypeScript, and PyTorch, '
    'deployed on Hugging Face Spaces with Docker.')
pdf.ln(1)
pdf.set_font('Times', 'B', 11); pdf.set_text_color(0,51,102)
pdf.cell(0, 6, 'Keywords:', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Times', 'I', 10); pdf.set_text_color(80,80,80)
pdf.multi_cell(0, 5, 'NCF, Deep Learning, Cosine Similarity, Emotion Detection, Multi-Modal Search, Explainable AI')

# SECTION 2: INTRODUCTION (on same page as abstract)
pdf.ln(3)
pdf.section_title('2', 'Introduction')
pdf.sub_title('2.1 Background')
pdf.body_text(
    'Recommendation systems have become essential in modern digital platforms. In the book domain, '
    'with millions of titles available, readers face a paradox of choice. Machine learning-driven '
    'recommendation systems address this by learning from historical reading patterns to make '
    'personalized predictions, improving both user engagement and content discovery.')
pdf.sub_title('2.2 Problem Statement')
pdf.body_text('The key challenges addressed by BOOKIFY include:')
pdf.bullet('Data Sparsity: 99.48% sparse user-book matrix makes traditional methods unreliable')
pdf.bullet('Cold Start Problem: New users lack interaction data for accurate recommendations')
pdf.bullet('Explainability: Users need to understand why a book is recommended to build trust')
pdf.bullet('Multi-modal Discovery: Single-input systems limit the ways users can discover books')
pdf.sub_title('2.3 Objectives')
pdf.bullet('Design and train an NCF deep learning model for accurate book rating prediction')
pdf.bullet('Build dual-mode recommendations (Classic Cosine Similarity + AI-powered NCF)')
pdf.bullet('Implement genre-based, mood-based, and multi-modal recommendation capabilities')
pdf.bullet('Create explainable AI with transparent reasoning for every recommendation')
pdf.bullet('Deploy a production-ready web application with premium UI and security')


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DATASET & DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('3', 'Dataset & Data Analysis')
pdf.sub_title('3.1 Dataset Overview')
pdf.body_text(
    'The Book-Crossing dataset, collected by Cai-Nicolas Ziegler, contains 278,858 users, '
    '271,360 books, and 1,149,780 ratings. After filtering (users with 200+ ratings, books '
    'with 50+ ratings), the working dataset comprises:')
pdf.kv_row('Total Books', f'{acc["dataset"]["total_books"]:,}', True)
pdf.kv_row('Total Users', f'{acc["dataset"]["total_users"]:,}', True)
pdf.kv_row('Total Ratings', f'{acc["dataset"]["total_ratings"]:,}', True)
pdf.kv_row('Rating Range', f'{acc["dataset"]["rating_range"][0]} - {acc["dataset"]["rating_range"][1]}', True)
pdf.kv_row('Genre Categories', str(len(genre_counts)), True)
pdf.kv_row('Matrix Sparsity', f'{sp:.2f}%', True)

pdf.ln(2)
pdf.sub_title('3.2 User-Book Matrix Sparsity')
pdf.add_chart(f'{OUT_DIR}/sparsity.png', 135)

pdf.sub_title('3.3 Genre Distribution')
pdf.add_chart(f'{OUT_DIR}/genre_distribution.png', 160)

pdf.add_page()
pdf.sub_title('3.4 Most Popular Books')
pdf.add_chart(f'{OUT_DIR}/top_books_votes.png', 160)

pdf.sub_title('3.5 Rating Distribution')
pdf.add_chart(f'{OUT_DIR}/rating_distribution.png', 135)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: METHODOLOGY
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('4', 'Methodology - NCF Model')
pdf.sub_title('4.1 Architecture')
pdf.bullet('Embedding: 64-dim for users and books -> Concat(128) -> MLP[128,64,32] -> Sigmoid')
pdf.bullet(f'Total parameters: {acc["total_parameters"]:,}')
pdf.add_chart(f'{OUT_DIR}/architecture.png', 160)

pdf.sub_title('4.2 Training Configuration')
pdf.kv_row('Optimizer', 'Adam (lr=0.001)')
pdf.kv_row('Loss', 'BCE | Batch 512 | Early Stop epoch {}'.format(acc['training']['epochs_run']))
pdf.kv_row('Split', '70% train / 15% val / 15% test')
pdf.kv_row('Training Time', f'{acc["training_time_seconds"]}s')
pdf.ln(1)
pdf.add_chart(f'{OUT_DIR}/dataset_split.png', 100)

pdf.sub_title('4.3 Training Progress')
pdf.body_text(
    f'Training converged at epoch {acc["training"]["epochs_run"]} with train loss '
    f'{acc["training"]["final_train_loss"]:.4f} and val loss {acc["training"]["best_val_loss"]:.4f}.')
pdf.add_chart(f'{OUT_DIR}/training_loss.png', 145)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: RESULTS & PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('5', 'Results & Performance Evaluation')
pdf.add_chart(f'{OUT_DIR}/performance_metrics.png', 170)
pdf.sub_title('5.1 Prediction Metrics')
pdf.kv_row('Accuracy', f'{acc["accuracy"]["accuracy_pct"]}%', True)
pdf.kv_row('RMSE / MAE', f'{acc["accuracy"]["rmse"]} / {acc["accuracy"]["mae"]}', True)
pdf.kv_row('NRMSE', str(acc['accuracy']['nrmse']), True)
pdf.ln(2)
pdf.sub_title('5.2 Ranking Metrics')
pdf.kv_row('Precision@10', f'{acc["ranking"]["precision_at_10"]:.4f}', True)
pdf.kv_row('NDCG@10 / Hit Rate@10', f'{acc["ranking"]["ndcg_at_10"]:.4f} / {acc["ranking"]["hit_rate_at_10"]:.4f}', True)
pdf.ln(2)
pdf.sub_title('5.3 Classic vs NCF Comparison')
pdf.add_chart(f'{OUT_DIR}/classic_vs_ncf.png', 140)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: SYSTEM ARCHITECTURE & FEATURES
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('6', 'System Architecture & Features')
pdf.bullet('Backend: Flask (1168 lines, 20+ routes), PyTorch, SQLite, Clerk Auth')
pdf.bullet('Frontend: React 18 + TypeScript + Vite + Framer Motion')
pdf.bullet('ML: NCF model, cosine similarity, genre classifier, face-api.js')
pdf.bullet('Deployment: Docker on Hugging Face Spaces + Render')
pdf.ln(2)
features = [
    ('Dual Mode', 'Classic cosine vs AI NCF engines'),
    ('Smart Search', 'Fuzzy matching + autocomplete (4,659 books)'),
    ('Genre Explorer', '23 categories, centrality-based ranking'),
    ('Mood Detection', 'Webcam emotion -> genre mapping'),
    ('Multi-Modal', '5 channels: text, voice, image, mood, history'),
    ('Explainable AI', 'Similarity %, genre, author, popularity reasons'),
    ('Wishlist/History', 'Persistent collections with SQLite'),
    ('For You', 'Personalized from reading history + onboarding'),
]
for name, desc in features:
    pdf.set_font('Times', 'B', 10); pdf.set_text_color(0,51,102)
    pdf.cell(40, 5.5, name)
    pdf.set_font('Times', '', 9.5); pdf.set_text_color(60,60,60)
    pdf.cell(0, 5.5, desc, new_x="LMARGIN", new_y="NEXT")

# SECTION 7: TECH STACK (on same page)
pdf.ln(4)
pdf.section_title('7', 'Technology Stack')
pdf.kv_row('Backend', 'Python 3.10+, Flask 3.x, PyTorch 2.x, SQLite')
pdf.kv_row('Frontend', 'React 18, TypeScript, Vite 5, Framer Motion')
pdf.kv_row('ML/AI', 'scikit-learn, face-api.js, Web Speech API')
pdf.kv_row('Deploy', 'Docker, Hugging Face Spaces, Render, Git LFS')
pdf.kv_row('Auth', 'Clerk SDK (OAuth 2.0)')


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: SCREENSHOTS
# ═══════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_title('8', 'Application Screenshots')

screenshots = [
    ('Homepage', 'ss_homepage.png'),
    ('Recommendation Engine', 'ss_recommend.png'),
    ('Search Results', 'ss_results.png'),
    ('Mood-Based Reader', 'ss_mood.png'),
    ('Book Detail Modal', 'ss_modal.png'),
    ('User Profile', 'ss_profile.png'),
]
for title, filename in screenshots:
    ss_path = f'{OUT_DIR}/{filename}'
    if os.path.exists(ss_path):
        pdf.set_font('Times', 'B', 11); pdf.set_text_color(0,51,102)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
        pdf.add_chart(ss_path, 115)
        if pdf.get_y() > 235:
            pdf.add_page()


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: CONCLUSION
# ═══════════════════════════════════════════════════════════════════
pdf.section_title('9', 'Conclusion & Future Work')
pdf.body_text(
    f'BOOKIFY achieves {acc["accuracy"]["accuracy_pct"]}% accuracy with RMSE {acc["accuracy"]["rmse"]} on '
    f'{acc["dataset"]["total_ratings"]:,} interactions across {acc["dataset"]["total_books"]:,} books. '
    'The platform offers 14+ features including dual recommendation modes, mood-based search, '
    'explainable AI, and is deployed on Hugging Face Spaces with Docker.')
pdf.sub_title('Future Enhancements')
pdf.bullet('Transformer models (BERT) for semantic understanding')
pdf.bullet('Implicit feedback and real-time model retraining')
pdf.bullet('Social features and reinforcement learning for diversity')

pdf.ln(4)
# ═══════════════════════════════════════════════════════════════════
# SECTION 10: REFERENCES
# ═══════════════════════════════════════════════════════════════════
pdf.section_title('10', 'References')
refs = [
    '[1] He, X. et al. (2017). "Neural Collaborative Filtering." WWW, pp. 173-182.',
    '[2] Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems." IEEE Computer, 42(8), pp. 30-37.',
    '[3] Ziegler, C.N. et al. (2005). "Improving Recommendation Lists Through Topic Diversification." WWW, pp. 22-32.',
    '[4] Burke, R. (2002). "Hybrid Recommender Systems: Survey and Experiments." UMUAI, 12(4), pp. 331-370.',
    '[5] Zhang, Y. & Chen, X. (2020). "Explainable Recommendation: A Survey." FnTIR, 14(1), pp. 1-101.',
    '[6] Goodfellow, I. et al. (2016). "Deep Learning." MIT Press.',
    '[7] Kingma, D.P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." ICLR.',
    '[8] Flask Documentation. https://flask.palletsprojects.com/',
    '[9] PyTorch Documentation. https://pytorch.org/docs/',
    '[10] face-api.js. https://github.com/justadudewhohacks/face-api.js',
    '[11] Open Library API. https://openlibrary.org/developers/api',
    '[12] Book-Crossing Dataset. https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset',
]
for ref in refs:
    pdf.set_font('Times', '', 10.5); pdf.set_text_color(40,40,40)
    pdf.multi_cell(0, 5.5, ref)
    pdf.ln(2)


# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
output_path = 'BOOKIFY_Project_Report.pdf'
pdf.output(output_path)
print(f"\nReport saved: {output_path}")
print(f"   Pages: {pdf.page_no()}")
print(f"   Charts: 10")
print(f"   Size: {os.path.getsize(output_path) / 1024:.0f} KB")

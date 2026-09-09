"""Generate the GRADE architecture diagram as draw.io (mxGraph) XML.

Fixes over the generated draft:
  * `as='geometry'` is a Python syntax error (`as` is a reserved keyword);
    attributes are passed via a dict instead.
  * Edges anchored to background containers ("b3_bg") are re-anchored to
    real nodes, otherwise draw.io draws them to the panel corner.
  * The decorative rotated "cross" shape overlapped the rejected-corpus
    cards and made them unreadable; removed (each card is struck through
    individually instead).
  * Numbers are taken from the measured audit, not restated by hand:
    see data/corpus_stats.csv and data/baselines.csv.

Output: paper/grade_architecture.drawio  (File > Open From > Device in
diagrams.net, or drag-and-drop onto the canvas).
"""
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'grade_architecture.drawio')

# ---------------------------------------------------------------- styles
HDR = ('shape=rectangle;fillColor={c};strokeColor=none;fontColor=#FFFFFF;'
       'fontStyle=1;fontSize=13;whiteSpace=wrap;html=1;align=center;'
       'verticalAlign=middle;rounded=1;')
BG = ('rounded=1;fillColor={f};strokeColor={s};strokeWidth=1.5;'
      'whiteSpace=wrap;html=1;verticalAlign=top;spacingTop=40;')
CARD = ('rounded=1;fillColor=#FFFFFF;strokeColor=#BDC3C7;strokeWidth=1;'
        'whiteSpace=wrap;html=1;fontSize=11;fontColor=#2C3E50;'
        'align=center;verticalAlign=middle;')
CARD_B = ('rounded=1;fillColor=#FFFFFF;strokeColor=#34495E;strokeWidth=2;'
          'whiteSpace=wrap;html=1;fontSize=11;fontStyle=1;fontColor=#2C3E50;'
          'align=center;verticalAlign=middle;')
REJ = ('rounded=1;fillColor=#FDEDEC;strokeColor=#E6B0AA;strokeWidth=1;'
       'dashed=1;fontColor=#7F8C8D;whiteSpace=wrap;html=1;fontSize=10;'
       'align=center;verticalAlign=middle;')
TAG = ('shape=note;whiteSpace=wrap;html=1;size=12;fillColor=#FFF9C4;'
       'strokeColor=#FBC02D;fontSize=9;fontColor=#5D4037;align=left;'
       'spacingLeft=6;verticalAlign=middle;')
FORMULA = ('rounded=1;fillColor=#EAEDED;strokeColor=#95A5A6;fontSize=10;'
           'fontStyle=1;fontColor=#2C3E50;align=center;verticalAlign=middle;')
LABEL = ('text;html=1;align=center;verticalAlign=middle;fontSize=11;'
         'fontStyle=1;fontColor={c};')

COLORS = {
    1: ('#1E6CDB', '#F2F7FF'),
    2: ('#6B3BA7', '#F7F2FF'),
    3: ('#E67E22', '#FFF8F0'),
    4: ('#16A085', '#F0F9F8'),
    5: ('#2C3E50', '#F8F9FA'),
}

root = None


def node(nid, value, style, x, y, w, h):
    c = ET.SubElement(root, 'mxCell', id=nid, value=value, style=style,
                      vertex='1', parent='1')
    ET.SubElement(c, 'mxGeometry', {'x': str(x), 'y': str(y),
                                    'width': str(w), 'height': str(h),
                                    'as': 'geometry'})
    return c


def edge(eid, src, tgt, style='', value=''):
    base = ('edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;'
            'endArrow=block;endFill=1;')
    c = ET.SubElement(root, 'mxCell', id=eid, value=value,
                      style=base + style, edge='1', parent='1',
                      source=src, target=tgt)
    ET.SubElement(c, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
    return c


def panel(n, title, x, y, w, h):
    stroke, fill = COLORS[n]
    node(f'b{n}_bg', '', BG.format(f=fill, s=stroke), x, y, w, h)
    node(f'b{n}_hd', title, HDR.format(c=stroke), x, y, w, 34)


def build():
    global root
    mx = ET.Element('mxfile', host='app.diagrams.net', type='device')
    dia = ET.SubElement(mx, 'diagram', id='grade', name='GRADE Architecture')
    gm = ET.SubElement(dia, 'mxGraphModel', dx='1600', dy='900', grid='1',
                       gridSize='10', guides='1', tooltips='1', connect='1',
                       arrows='1', fold='1', page='1', pageScale='1',
                       pageWidth='1700', pageHeight='780', math='0', shadow='0')
    root = ET.SubElement(gm, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')

    small = "<span style='font-size:9px;color:#555;'>"

    # ---------------------------------------------------- 1. CORPORA
    panel(1, '1. CORPORA / INPUT', 20, 20, 330, 700)
    node('b1_t1', 'Used in study', LABEL.format(c='#1E6CDB'), 30, 62, 150, 20)
    node('b1_t2', 'Rejected after audit', LABEL.format(c='#C0392B'), 190, 62, 150, 20)

    node('c_llm', f"<b>LLM-Detect</b><br>{small}27,226 essays<br>"
         f"16,042 human / 11,184 AI<br>real student writing<br>"
         f"2 assigned prompts</span><br>"
         f"<b style='color:#1E6CDB;font-size:9px;'>TRAIN / VAL / TEST</b>",
         CARD, 30, 88, 150, 130)
    node('c_pile', f"<b>AI Text Pile</b><br>{small}20,000 items<br>"
         f"10,000 / 10,000<br>MIT licensed<br>web + chatbot text</span><br>"
         f"<b style='color:#E67E22;font-size:9px;'>HELD OUT</b>",
         CARD, 30, 232, 150, 125)
    node('c_abs', f"<b>Abstract Paraphrase</b><br>{small}304 items<br>"
         f"152 / 152 pairs<br>PhD abstracts<br>GPT-4, temp 0</span><br>"
         f"<b style='color:#E67E22;font-size:9px;'>HELD OUT</b>",
         CARD, 30, 371, 150, 125)

    strike = "<span style='text-decoration:line-through;'><b>{}</b></span>"
    tiny = "<span style='font-size:9px;color:#7F8C8D;'>"
    node('r_aide', strike.format('AIDE') + f"<br>{tiny}1,378 rows<br>"
         f"100% subset of<br>LLM-Detect; 3 AI rows</span>", REJ, 190, 88, 150, 95)
    node('r_sa', strike.format('Synthetic A') + f"<br>{tiny}10,240 rows<br>"
         f"label printed<br>literally in text</span>", REJ, 190, 193, 150, 95)
    node('r_sb', strike.format('Synthetic B') + f"<br>{tiny}15,000 rows<br>"
         f"both classes from<br>same 300 sentences</span>", REJ, 190, 298, 150, 95)
    node('r_nlp', strike.format('nlpdataset') + f"<br>{tiny}690 rows<br>"
         f"off-task<br>(news sentences)</span>", REJ, 190, 403, 150, 93)

    node('b1_audit', "<b>Audit outcome</b><br>"
         "<span style='font-size:9px;'>4 of 6 candidate corpora "
         "unusable for detector evaluation</span>",
         TAG, 190, 515, 150, 60)

    # ---------------------------------------------------- 2. PROTOCOL
    panel(2, '2. GRADE PROTOCOL', 370, 20, 340, 700)
    node('b2_sub', "<span style='font-size:9px;color:#6B3BA7;'><b>GRADE</b> = "
         "Generalization Robustness<br>Assessment for Detectors in Education</span>",
         'text;html=1;align=center;verticalAlign=middle;', 380, 58, 320, 26)

    # Mirrors the LOGO block: one operation box on top, then three rows.
    # LOGO rotated the held-out generator; here the training corpus is
    # FIXED and the evaluation corpus varies.
    node('p_fix', "<b>Fix the training corpus</b><br>"
         f"{tiny}train once, then vary the evaluation corpus</span>",
         CARD_B, 400, 92, 280, 46)

    ROW = ('rounded=1;fillColor=#FFFFFF;strokeColor=#6B3BA7;strokeWidth=1.5;'
           'whiteSpace=wrap;html=1;fontSize=10;fontColor=#2C3E50;'
           'align=center;verticalAlign=middle;')
    TGT = ('rounded=1;fillColor=#F7F2FF;strokeColor=#BDC3C7;strokeWidth=1;'
           'whiteSpace=wrap;html=1;fontSize=10;fontColor=#2C3E50;'
           'align=center;verticalAlign=middle;')

    rows = [
        ('a', 'Eval A &mdash; train on LLM-Detect', 'in-distribution test', 170),
        ('b', 'Eval B &mdash; train on LLM-Detect', 'AI Text Pile', 240),
        ('c', 'Eval C &mdash; train on LLM-Detect', 'Abstract Paraphrase', 310),
    ]
    for tag, left, right, yy in rows:
        node(f'row_{tag}', f"<b>{left}</b>", ROW, 390, yy, 160, 48)
        node(f'tgt_{tag}', f"<b>{right}</b>", TGT, 588, yy, 108, 48)
        lbl = 'test' if tag == 'a' else 'test, unseen'
        edge(f'ed_{tag}', f'row_{tag}', f'tgt_{tag}',
             'strokeColor=#6B3BA7;dashed=1;', lbl)

    node('p_gap', 'generalization gap  =  in-distribution '
         '&minus; held-out accuracy', FORMULA, 395, 392, 300, 34)

    node('p_note', "<b>protocol guarantees</b><br>"
         "<span style='font-size:9px;'>&bull; deduplicated: 1,919 duplicate "
         "texts removed<br>"
         "&bull; grouped 60/20/20; audit <b>raises</b> on any overlap<br>"
         "&bull; every split balanced &rarr; chance = 50%<br>"
         "&bull; validation used for checkpoint selection only<br>"
         "&bull; held-out corpora touched <b>once</b>, at inference</span>",
         TAG, 390, 444, 306, 104)

    edge('e1', 'c_llm', 'p_fix', 'strokeColor=#1E6CDB;strokeWidth=2;')
    edge('e2', 'p_fix', 'row_a', 'strokeColor=#6B3BA7;')
    edge('e3', 'p_fix', 'row_b', 'strokeColor=#6B3BA7;')
    edge('e4', 'p_fix', 'row_c', 'strokeColor=#6B3BA7;')

    # ---------------------------------------------------- 3. BACKBONES
    panel(3, '3. ENCODER BACKBONES (5 variants)', 730, 20, 280, 700)
    node('b3_note', "<span style='font-size:9px;color:#E67E22;'>identical recipe "
         "for all five, so any difference<br>is attributable to architecture</span>",
         'text;html=1;align=center;verticalAlign=middle;', 740, 58, 260, 28)

    models = [('BERT-base-uncased', '~110M'), ('RoBERTa-base', '~125M'),
              ('DeBERTa-v3-small', '~140M'), ('DistilBERT-base-uncased', '~66M'),
              ('ELECTRA-base-discriminator', '~110M')]
    y = 92
    for i, (m, p) in enumerate(models, 1):
        node(f'm{i}', f"<b>{m}</b><br>{tiny}{p} parameters</span>",
             CARD, 750, y, 240, 44)
        y += 52

    node('b3_cfg', "<b>training recipe</b><br>"
         "<span style='font-size:9px;'>max length 256 &nbsp;|&nbsp; AdamW<br>"
         "lr 2&times;10<sup>-5</sup> &nbsp;|&nbsp; batch size 8<br>"
         "5 epochs &nbsp;|&nbsp; cross-entropy<br>"
         "full fine-tuning (encoder + head)<br>"
         "best-val-accuracy checkpoint<br>"
         "seed 42 (single seed)</span>", TAG, 750, 372, 240, 108)

    edge('e15', 'p_fix', 'm3', 'strokeColor=#E67E22;strokeWidth=2;')

    # ---------------------------------------------------- 4. CONTROLS
    panel(4, '4. CONTROL BASELINES', 1030, 20, 280, 700)
    node('b4_len', "<b>length-only classifier</b><br>"
         f"{small}one feature: document length in characters</span>",
         CARD, 1050, 66, 240, 52)
    node('b4_tf', "<b>TF-IDF classifier</b><br>"
         f"{small}word unigrams + bigrams</span>", CARD, 1050, 130, 240, 52)
    node('b4_fit', "<span style='font-size:9px;color:#16A085;'><i>fitted "
         "independently on every evaluation corpus</i></span>",
         'text;html=1;align=center;verticalAlign=middle;', 1050, 190, 240, 24)

    node('b4_claim', "<i>A detector that does not clearly exceed these has "
         "not demonstrated detection on that corpus.</i>",
         'rounded=1;fillColor=#E8F8F5;strokeColor=#16A085;fontSize=9;'
         'fontColor=#117A65;align=center;verticalAlign=middle;spacing=6;'
         'whiteSpace=wrap;html=1;', 1050, 220, 240, 52)

    node('b4_meas', "<b>measured</b><br>"
         "<span style='font-size:9px;'>length-only reaches <b>80.3%</b> on the "
         "AI Text Pile<br>but only <b>54.5%</b> on LLM-Detect.<br><br>"
         "The confound is a property of how a<br>corpus was assembled, not of "
         "the task.</span>", TAG, 1050, 286, 240, 96)

    node('b4_tf2', "<span style='font-size:9px;color:#7F8C8D;'>TF-IDF reaches "
         "98.6% in distribution: a transformer<br>scoring in that range has not "
         "improved on<br>bag-of-words.</span>",
         'rounded=1;fillColor=#FFFFFF;strokeColor=#D5DBDB;fontSize=9;'
         'align=center;verticalAlign=middle;whiteSpace=wrap;html=1;',
         1050, 396, 240, 60)

    edge('e16', 'tgt_b', 'b4_len', 'strokeColor=#16A085;strokeWidth=2;')

    # ---------------------------------------------------- 5. OUTPUTS
    panel(5, '5. OUTPUTS', 1330, 20, 280, 700)
    node('o_a', "<b>5A. Metrics</b><br>"
         f"{small}accuracy, precision, <b>recall</b>, F1,<br>"
         f"ROC-AUC, FPR, <b>FNR</b></span><br>"
         f"{tiny}recall and FNR emphasised: a missed<br>AI essay is the failure "
         f"the control was<br>adopted to prevent</span>",
         CARD, 1350, 66, 240, 110)
    node('o_b', "<b>5B. Generalization gap</b><br>"
         f"{small}in-distribution vs held-out,<br>per architecture</span><br>"
         f"{tiny}does in-distribution ranking predict<br>held-out ranking?</span>",
         CARD, 1350, 190, 240, 92)
    node('o_c', "<b>5C. Control comparison</b><br>"
         f"{small}every detector scored against the<br>"
         f"length-only and TF-IDF baselines,<br>per corpus</span>",
         CARD, 1350, 296, 240, 80)
    node('o_d', "<b>5D. Evidence requirements</b><br>"
         f"{small}what an institution should require<br>"
         f"before adopting a detector as an<br>assessment control</span>",
         CARD, 1350, 390, 240, 80)

    edge('e17', 'b4_meas', 'o_a', 'strokeColor=#2C3E50;strokeWidth=2;')
    edge('e18', 'b4_meas', 'o_b', 'strokeColor=#2C3E50;')
    edge('e19', 'b4_meas', 'o_c', 'strokeColor=#2C3E50;')

    xml = ET.tostring(mx, encoding='utf-8')
    pretty = minidom.parseString(xml).toprettyxml(indent='  ')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(pretty)
    return pretty


if __name__ == '__main__':
    out = build()
    print(f'wrote {OUT}')
    print(f'{len(out)} chars, {out.count("<mxCell")} cells')

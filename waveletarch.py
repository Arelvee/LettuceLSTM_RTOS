from graphviz import Digraph

dot = Digraph(comment='Sensor-Specific Wavelet Processing Pipeline', format='pdf')
dot.attr(rankdir='TB', bgcolor='white', splines='ortho')
dot.attr('node', shape='plaintext', fontname='Helvetica', fontsize='12')

# Sensor parameters table
sensor_table = '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR>
    <TD BGCOLOR="#f0f0f0"><B>Sensor</B></TD>
    <TD BGCOLOR="#f0f0f0"><B>Wavelet</B></TD>
    <TD BGCOLOR="#f0f0f0"><B>Level</B></TD>
    <TD BGCOLOR="#f0f0f0"><B>Threshold Mode</B></TD>
  </TR>
  <TR><TD>humidity</TD><TD>db8</TD><TD>5</TD><TD>soft</TD></TR>
  <TR><TD>temp_envi</TD><TD>db4</TD><TD>3</TD><TD>hard</TD></TR>
  <TR><TD>temp_water</TD><TD>db4</TD><TD>3</TD><TD>hard</TD></TR>
  <TR><TD>tds</TD><TD>db4</TD><TD>4</TD><TD>soft</TD></TR>
  <TR><TD>ec</TD><TD>db4</TD><TD>4</TD><TD>soft</TD></TR>
  <TR><TD>lux</TD><TD>db4</TD><TD>4</TD><TD>soft</TD></TR>
  <TR><TD>ppfd</TD><TD>db4</TD><TD>4</TD><TD>soft</TD></TR>
  <TR><TD>reflect_445</TD><TD>sym8</TD><TD>6</TD><TD>garrote</TD></TR>
  <TR><TD>reflect_480</TD><TD>sym8</TD><TD>6</TD><TD>garrote</TD></TR>
  <TR><TD>ph</TD><TD>db4</TD><TD>4</TD><TD>soft</TD></TR>
</TABLE>>'''

dot.node('params', sensor_table, shape='plaintext')

# Processing pipeline nodes with detailed functions
dot.node('RC', 'R C\nRaw Sensor Input & Cleaning\n(Per-sensor calibration)')
dot.node('S1', '∑\nSignal Decomposition\n(Multi-level wavelet decomposition)')
dot.node('G1', 'G₁\nWavelet Transform\n(Discrete Wavelet Transform)\n• Daubechies (db4/db8)\n• Symlet (sym8)')
dot.node('G2', 'G₂\nThresholding\n(Noise reduction)\n• Soft thresholding\n• Hard thresholding\n• Garrote thresholding')
dot.node('S2', '∑\nSignal Reconstruction\n(Inverse DWT)')
dot.node('OUT', 'Output for Machine Learning\nReconstructed Signal\n(Cleaned time-series data)')

# Define connections
connections = [
    ('RC', 'S1'),
    ('S1', 'G1'), 
    ('S1', 'G2'),
    ('G1', 'S2'), 
    ('G2', 'S2'),
    ('S2', 'OUT')
]

# Add straight edges
for start, end in connections:
    dot.edge(start, end, arrowhead='normal', color='black')

# Add parameter references
dot.edge('params', 'G1', style='dashed', color='gray', label='Wavelet/Level', fontsize='10')
dot.edge('params', 'G2', style='dashed', color='gray', label='Threshold Mode', fontsize='10')

# Render the diagram
dot.render('sensor_wavelet_pipeline', cleanup=True, view=True)
"""
PDF Report Generator for Banking Copilot.
Generates professional briefing packs using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

class ReportGenerator:
    """
    Generates PDF reports for country banking assessments.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Define custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1E88E5')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#0E1117'),
            borderPadding=(0, 0, 5, 0),
            borderWidth=0,
            borderColor=colors.HexColor('#1E88E5')
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskBadge',
            fontSize=12,
            textColor=colors.white,
            backColor=colors.HexColor('#1E88E5'),
            alignment=1, # Center
            borderPadding=5,
            borderRadius=4
        ))

    def generate_report(self, country_name: str, country_code: str, 
                       scores: dict, insights: dict) -> bytes:
        """
        Generate a PDF report in memory and return bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )
        
        story = []
        
        # --- Header ---
        story.append(Paragraph(f"Banking System Scorecard: {country_name}", self.styles['ReportTitle']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y')}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1E88E5')))
        story.append(Spacer(1, 20))
        
        # --- Executive Summary Score ---
        # Create a nice summary table
        data = [
            ['Component', 'Score', 'Rating'],
            ['Composite Score', f"{scores['composite']:.1f}", scores['tier']],
            ['Economic Resilience', f"{scores['economic']:.1f}", 'n/a'],
            ['Industry Risk', f"{scores['industry']:.1f}", 'n/a']
        ]
        
        t = Table(data, colWidths=[200, 100, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E88E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F6F9')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # --- Key Strengths ---
        story.append(Paragraph("Key Strengths", self.styles['SectionHeader']))
        for s in insights.get('key_strengths', []):
            # Bulllet point style
            p = Paragraph(f"<font color='green'>&#10003;</font> {s}", self.styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 6))
            
        # --- Key Risks ---
        story.append(Paragraph("Key Risks", self.styles['SectionHeader']))
        for r in insights.get('key_risks', []):
            p = Paragraph(f"<font color='red'>&#9888;</font> {r}", self.styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 6))
            
        # --- Detailed Analysis (Banking) ---
        story.append(Paragraph("Banking Sector Analysis", self.styles['SectionHeader']))
        for note in insights.get('banking_sector', []):
            story.append(Paragraph(note, self.styles['Normal']))
            story.append(Spacer(1, 8))
            
        # --- Detailed Analysis (Macro) ---
        story.append(Paragraph("Macroeconomic Context", self.styles['SectionHeader']))
        for note in insights.get('macro_context', []):
            story.append(Paragraph(note, self.styles['Normal']))
            story.append(Spacer(1, 8))
            
        # Build
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

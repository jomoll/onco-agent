"""
Mappings from common shorthand/translated report type names to the canonical
report categories supported by ReportsRAGTool.
"""

REPORT_TYPE_SYNONYMS = {
    # tumor board decisions
    "beschluss": "tumor_board",
    "tumorboard": "tumor_board",
    "tumor board": "tumor_board",
    "tumorboard beschluss": "tumor_board",
    "tumorboardbeschluss": "tumor_board",
    "mtb": "tumor_board",
    "tumorkonferenz": "tumor_board",
    "tumorkonferenz beschluss": "tumor_board",
    "onkoboard": "tumor_board",

    # cytology
    "cytology": "cytology",
    "zytologie": "cytology",
    "zytologischer befund": "cytology",
    "zytologiebericht": "cytology",
    "cytology report": "cytology",

    # flow / immunophenotyping
    "flow": "flow",
    "flow cytometry": "flow",
    "flowcytometry": "flow",
    "immunophenotyping": "flow",
    "immunoekotypisierung": "flow",
    "immunphänotypisierung": "flow",
    "durchflusszytometrie": "flow",

    # pathology
    "path": "pathology",
    "pathology": "pathology",
    "pathology report": "pathology",
    "histology": "pathology",
    "histologie": "pathology",
    "histologischer befund": "pathology",
    "biopsy report": "pathology",
    "biopsiebericht": "pathology",
    "stanze": "pathology",
    "stanzbiopsie": "pathology",
    "gewebeuntersuchung": "pathology",

    # radiology
    "radiology": "radiology",
    "rad": "radiology",
    "radiology report": "radiology",
    "radiologischer befund": "radiology",
    "radiologiebericht": "radiology",
    "radiology note": "radiology",
    "ct bericht": "radiology",
    "mr bericht": "radiology",
    "mrt bericht": "radiology",
    "röntgenbericht": "radiology",
    "ultraschallbericht": "radiology",
    "sono bericht": "radiology",
    "befund ct": "radiology",
    "befund mrt": "radiology",
    "befund röntgen": "radiology",
    "sonographie": "radiology",

    # cardiology / echo
    "echokardiograp": "cardiology",
    "echo": "cardiology",
    "kardiologie": "cardiology",

    # consults
    "konsiliarbericht": "consult",
    "konsilbericht": "consult",
    "konsil": "consult",
    "konsiliar": "consult",

    # doctor letters
    "arztbrief": "doctor_letter",
    "arztbriefe": "doctor_letter",
    "entlassungsbrief": "doctor_letter",
    "entlassungsbericht": "doctor_letter",
    "aufnahmebrief": "doctor_letter",
    "aufnahmebericht": "doctor_letter",
    "verlaufsbericht": "doctor_letter",
    "arztbericht": "doctor_letter",
    "bericht": "doctor_letter",

    # history/anamnese
    "anamnese": "history",
    "anamnesebogen": "history",
    "basisanamnese": "history",
    "barthel index": "history",
}

import sqlite3
from datetime import datetime, timedelta
import json

def create_database():
    # Connect to database (creates if doesn't exist)
    conn = sqlite3.connect('clinical_rag.db')
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dob TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            encounter_id TEXT,
            doc_type TEXT NOT NULL,
            section TEXT,
            date TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            author TEXT,
            department TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')
    
    # Create FTS5 virtual table for full-text search
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            id UNINDEXED,
            title,
            content,
            content=documents,
            content_rowid=rowid
        )
    ''')
    
    # Create triggers to maintain FTS index
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_fts_insert AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, id, title, content) 
            VALUES (new.rowid, new.id, new.title, new.content);
        END
    ''')
    
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_fts_delete AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, id, title, content) 
            VALUES('delete', old.rowid, old.id, old.title, old.content);
        END
    ''')
    
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS documents_fts_update AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, id, title, content) 
            VALUES('delete', old.rowid, old.id, old.title, old.content);
            INSERT INTO documents_fts(rowid, id, title, content) 
            VALUES (new.rowid, new.id, new.title, new.content);
        END
    ''')
    
    print("Database tables created successfully!")
    return conn, cursor

def seed_database(conn, cursor):
    # Clear existing data
    cursor.execute('DELETE FROM documents')
    cursor.execute('DELETE FROM patients')
    
    # Insert patients
    patients = [
        ("p001", "Jane Doe", "1970-03-12"),
        ("p002", "John Smith", "1962-11-05"),
        ("p003", "Samir Patel", "1988-07-22"),
    ]
    
    cursor.executemany('INSERT INTO patients (id, name, dob) VALUES (?, ?, ?)', patients)
    
    # Insert documents with realistic medical content
    documents = [
        # Jane Doe - Cardiovascular focus
        (
            "doc-001-001", "p001", "enc-001-2024-09-20", "Progress Note", "Assessment and Plan",
            "2024-09-20T14:30:00Z", "Cardiology Follow-up",
            "Patient reports improved exercise tolerance since starting lisinopril 10mg daily. Blood pressure well controlled at 125/78 mmHg. Denies chest pain or dyspnea at rest. Current medications: Lisinopril 10mg daily, metoprolol 25mg BID. Patient reports good medication adherence. Continue current ACE inhibitor therapy. Recommend lifestyle modifications including low sodium diet (<2g/day) and regular aerobic exercise 150 minutes per week. Next follow-up appointment in 3 months. Will recheck basic metabolic panel to monitor renal function and electrolytes.",
            "Dr. Sarah Chen, MD", "Cardiology"
        ),
        (
            "doc-001-002", "p001", "enc-001-2024-08-15", "Lab Result", None,
            "2024-08-15T09:00:00Z", "Comprehensive Metabolic Panel",
            "Blood Urea Nitrogen (BUN): 18 mg/dL (normal range 7-20), Creatinine: 1.0 mg/dL (normal range 0.6-1.2), estimated GFR: >60 mL/min/1.73m² (normal), Potassium: 4.2 mEq/L (normal range 3.5-5.0), Sodium: 140 mEq/L (normal), Chloride: 102 mEq/L (normal). Renal function remains stable on ACE inhibitor therapy. No electrolyte abnormalities detected. Safe to continue lisinopril at current dose.",
            "Clinical Laboratory", "Laboratory"
        ),
        (
            "doc-001-003", "p001", "enc-001-2024-07-10", "Radiology Report", "Impression",
            "2024-07-10T11:20:00Z", "Transthoracic Echocardiogram",
            "Left ventricular systolic function is normal with estimated ejection fraction of 60-65%. No regional wall motion abnormalities. Left atrial size is normal. Mild mitral regurgitation present, likely functional. Right ventricular size and function normal. No pericardial effusion. Aortic valve appears normal with no significant stenosis or regurgitation. Pulmonary artery pressure estimated at 25 mmHg (normal). Recommendation: Continue medical management of hypertension. Repeat echo in 2-3 years or sooner if symptoms develop.",
            "Dr. Michael Rodriguez, MD", "Cardiology"
        ),
        (
            "doc-001-004", "p001", "enc-001-2024-06-01", "Discharge Summary", None,
            "2024-06-01T16:45:00Z", "Hypertensive Crisis Management",
            "54-year-old female with history of essential hypertension admitted for hypertensive urgency. Admission blood pressure 185/115 mmHg. Patient had been non-adherent to antihypertensive medications for several weeks due to insurance issues. Started on lisinopril 5mg daily, gradually titrated to 10mg daily over 3 days with excellent blood pressure response. Added metoprolol 25mg BID on day 2. Discharge blood pressure 135/82 mmHg. Patient counseled extensively on medication adherence and lifestyle modifications. Social work consulted for insurance coverage issues. Follow-up arranged with cardiology in 1 week, then primary care in 2 weeks.",
            "Dr. Jennifer Park, MD", "Internal Medicine"
        ),
        
        # John Smith - Diabetes and complications
        (
            "doc-002-001", "p002", "enc-002-2024-09-18", "Progress Note", "Assessment and Plan",
            "2024-09-18T10:15:00Z", "Diabetes Management Visit",
            "62-year-old male with type 2 diabetes mellitus. HbA1c has improved significantly to 7.2% from previous 8.8% six months ago. Current medications: Metformin 1000mg twice daily, glipizide 10mg twice daily, started 3 months ago. Patient reports good medication adherence and has been following diabetic diet with nutritionist guidance. Reports 2-3 episodes of mild hypoglycemia per week, usually pre-meal, symptoms resolve with glucose tablets. Blood glucose logs show fasting glucose 120-150 mg/dL, post-prandial 160-200 mg/dL. Continue current regimen. Counseled on hypoglycemia management and carbohydrate counting. Next visit in 3 months with repeat HbA1c.",
            "Dr. Amanda Foster, MD", "Endocrinology"
        ),
        (
            "doc-002-002", "p002", "enc-002-2024-09-10", "Lab Result", None,
            "2024-09-10T08:30:00Z", "Diabetic Monitoring Panel",
            "HbA1c: 7.2% (target <7% for most adults), Fasting glucose: 145 mg/dL (goal <130), Microalbumin/creatinine ratio: 45 mg/g (elevated; normal <30), Creatinine: 1.1 mg/dL (normal), eGFR: 68 mL/min/1.73m² (mildly decreased), Total cholesterol: 195 mg/dL, LDL: 115 mg/dL (goal <100), HDL: 42 mg/dL (low), Triglycerides: 180 mg/dL (elevated). Microalbuminuria indicates early diabetic nephropathy. Recommend ACE inhibitor initiation for nephroprotection and statin therapy for cardiovascular risk reduction.",
            "Clinical Laboratory", "Laboratory"
        ),
        (
            "doc-002-003", "p002", "enc-002-2024-08-22", "Radiology Report", "Findings",
            "2024-08-22T13:40:00Z", "Chest X-ray PA and Lateral",
            "Heart size is within normal limits. Lungs are clear bilaterally with no evidence of infiltrate, effusion, or pneumothorax. No acute cardiopulmonary process identified. Costophrenic angles are sharp. No hilar lymphadenopathy. Bone structures appear normal. Clinical correlation: Diabetic patient with no evidence of pulmonary infection or cardiac complications on chest imaging.",
            "Dr. Lisa Wang, MD", "Radiology"
        ),
        (
            "doc-002-004", "p002", "enc-002-2024-07-15", "Progress Note", "Assessment and Plan",
            "2024-07-15T14:20:00Z", "Ophthalmology Diabetic Screening",
            "Annual diabetic retinal screening performed. Dilated fundoscopic examination reveals mild nonproliferative diabetic retinopathy in both eyes. Several microaneurysms and dot hemorrhages present in posterior pole bilaterally. No cotton wool spots or hard exudates. Macula appears normal with no evidence of diabetic macular edema. Optic discs normal. Visual acuity 20/20 both eyes. Recommend continued strict glycemic control to prevent progression. Annual follow-up recommended, sooner if vision changes occur.",
            "Dr. Robert Kim, MD", "Ophthalmology"
        ),
        
        # Samir Patel - Young adult with anxiety and respiratory issues  
        (
            "doc-003-001", "p003", "enc-003-2024-09-25", "Progress Note", "Assessment and Plan",
            "2024-09-25T11:00:00Z", "Primary Care Follow-up",
            "36-year-old male with history of generalized anxiety disorder and new respiratory symptoms. Reports significant improvement in anxiety symptoms since starting sertraline 50mg daily 8 weeks ago. No panic attacks in past month, sleep improved, work performance better. New chief complaint of intermittent dry cough and wheezing, especially with exercise and cold air exposure. No fever, no sputum production. Family history significant for asthma (mother). Physical exam: lungs clear to auscultation at rest, mild expiratory wheeze with forced expiration. Plan: Pulmonary function tests, chest CT to evaluate for asthma vs other causes. Continue sertraline. Albuterol inhaler prescribed for symptomatic relief.",
            "Dr. Maria Gonzalez, MD", "Family Medicine"
        ),
        (
            "doc-003-002", "p003", "enc-003-2024-09-25", "Lab Result", None,
            "2024-09-25T09:15:00Z", "Complete Blood Count with Differential",
            "White blood cells: 7,200/μL (normal range 4,500-11,000), Red blood cells: 4.8 M/μL (normal), Hemoglobin: 15.2 g/dL (normal), Hematocrit: 45% (normal), Platelets: 285,000/μL (normal), Neutrophils: 65% (normal), Lymphocytes: 28% (normal), Eosinophils: 5% (upper normal), Basophils: 1% (normal), Monocytes: 6% (normal). No evidence of infection, anemia, or hematologic abnormality. Slightly elevated eosinophils may suggest allergic component to respiratory symptoms.",
            "Clinical Laboratory", "Laboratory"
        ),
        (
            "doc-003-003", "p003", "enc-003-2024-09-23", "Radiology Report", "Impression",
            "2024-09-23T15:10:00Z", "High Resolution Chest CT",
            "Lungs demonstrate no focal consolidation, mass, or pleural effusion. No evidence of pulmonary embolism. Mild diffuse bronchial wall thickening noted throughout both lungs, more prominent in lower lobes. No honeycombing or traction bronchiectasis. Mediastinal and hilar lymph nodes normal size. Heart size normal. Findings consistent with reactive airway disease/asthma. No evidence of interstitial lung disease or malignancy. Recommend correlation with pulmonary function testing and response to bronchodilator therapy.",
            "Dr. Kevin Chen, MD", "Radiology"
        ),
        (
            "doc-003-004", "p003", "enc-003-2024-08-30", "Progress Note", "Assessment and Plan",
            "2024-08-30T16:30:00Z", "Psychiatry Follow-up Visit",
            "Generalized anxiety disorder showing excellent response to sertraline 50mg daily initiated 6 weeks ago. Patient reports marked improvement in anxiety symptoms, panic attacks have resolved, sleep quality much improved (now sleeping 7-8 hours vs previous 4-5 hours). Work performance and social functioning significantly better. No side effects from sertraline reported. PHQ-9 score decreased from 15 to 6. GAD-7 score improved from 18 to 8. Cognitive behavioral therapy sessions have been very helpful per patient report. Continue sertraline 50mg daily. Follow-up in 3 months or sooner if symptoms worsen.",
            "Dr. Emily Thompson, MD", "Psychiatry"
        ),
        (
            "doc-003-005", "p003", "enc-003-2024-08-15", "Lab Result", "Results",
            "2024-08-15T12:00:00Z", "Comprehensive Allergy Panel",
            "Environmental allergens tested via specific IgE levels: Dust mites (Dermatophagoides pteronyssinus): 15.2 kU/L (Class 4, high positive), Cat dander: 8.7 kU/L (Class 3, moderate positive), Ragweed: 6.1 kU/L (Class 3, moderate positive), Timothy grass: 4.2 kU/L (Class 2, low positive), Oak pollen: 3.8 kU/L (Class 2, low positive). Food allergens all negative including milk, egg, peanut, tree nuts, soy, wheat, shellfish. Total IgE elevated at 245 IU/mL (normal <100). Results indicate significant environmental allergies likely contributing to respiratory symptoms. Recommend environmental controls, allergen avoidance, and consideration of antihistamine therapy.",
            "Dr. Patricia Lee, MD", "Allergy/Immunology"
        )
    ]
    
    cursor.executemany('''
        INSERT INTO documents (id, patient_id, encounter_id, doc_type, section, date, title, content, author, department)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', documents)
    
    # Commit changes
    conn.commit()
    
    print(f"Database seeded successfully!")
    print(f"Inserted {len(patients)} patients and {len(documents)} documents")
    
    # Show some stats
    cursor.execute('SELECT COUNT(*) FROM patients')
    patient_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM documents')
    doc_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type')
    doc_types = cursor.fetchall()
    
    print(f"\nDatabase Statistics:")
    print(f"Total patients: {patient_count}")
    print(f"Total documents: {doc_count}")
    print(f"Document types:")
    for doc_type, count in doc_types:
        print(f"  {doc_type}: {count}")

def test_search(conn, cursor):
    print("\n" + "="*50)
    print("Testing search functionality:")
    
    # Test FTS search
    test_queries = [
        "blood pressure",
        "diabetes medication", 
        "anxiety sertraline",
        "chest pain",
        "lisinopril"
    ]
    
    for query in test_queries:
        cursor.execute('''
            SELECT d.title, d.doc_type, d.patient_id, 
                   snippet(documents_fts, 2, '[', ']', '...', 32) as snippet
            FROM documents_fts fts
            JOIN documents d ON d.rowid = fts.rowid  
            WHERE documents_fts MATCH ?
            ORDER BY bm25(documents_fts) 
            LIMIT 3
        ''', (query,))
        
        results = cursor.fetchall()
        print(f"\nQuery: '{query}' - Found {len(results)} results")
        for title, doc_type, patient_id, snippet in results:
            print(f"  {patient_id} | {doc_type} | {title}")
            print(f"    {snippet}")

if __name__ == "__main__":
    print("Creating Clinical RAG Database...")
    
    # Create and seed database
    conn, cursor = create_database()
    seed_database(conn, cursor)
    
    # Test search functionality
    test_search(conn, cursor)
    
    # Close connection
    conn.close()
    
    print("\nDatabase creation complete! File: clinical_rag.db")
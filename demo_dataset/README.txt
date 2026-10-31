================================================================================
CYBER GUARDIAN AI - EDUCATIONAL DEMONSTRATION DATASET
================================================================================

PROJECT:
    Cyber Guardian AI: AI-Powered Cyber Attacks and Defence
    Academic Prototype | Streamlit + Scikit-Learn Random Forest Pipeline

IMPORTANT DISCLAIMER:
    All records contained in this dataset are SYNTHETIC records generated 
    specifically for educational and academic demonstration purposes.
    Statistical characteristics reflect behavioral traffic patterns derived from
    standard cybersecurity benchmarks (e.g., CIC-IDS2017 / NSL-KDD), but they
    DO NOT represent live network packet captures, production corporate telemetry,
    or personally identifiable network information.

================================================================================
1. FILES IN THIS FOLDER
================================================================================

1. cyber_guardian_demo.csv
   - Description: Ready-to-upload prediction dataset.
   - Records: Exactly 1,000 network connection flows.
   - Columns: 20 feature columns (strictly matching REQUIRED_FEATURE_COLUMNS).
   - Target Label: Omitted (unlabelled), matching the application's prediction
     mode requirements (require_label=False).
   - Use Case: Direct upload into the Cyber Guardian AI "AI Threat Detection" page.

2. cyber_guardian_demo_labelled.csv
   - Description: Reference benchmark dataset containing ground truth labels.
   - Records: Exactly 1,000 network connection flows.
   - Columns: 21 columns (20 feature columns + 1 ground truth "label" column).
   - Use Case: Empirical model evaluation, retraining, confusion matrix validation,
     and multi-class performance comparison.

3. README.txt
   - This documentation file.

================================================================================
2. CLASS DISTRIBUTION (1,000 SAMPLES)
================================================================================

The 1,000 records are balanced across the 5 supported categories:

Category       Count    Percentage   Description
--------------------------------------------------------------------------------
BENIGN         600      60.0%        Normal operational network traffic flows
DoS            100      10.0%        Denial of Service (Volume/State Exhaustion)
DDoS           100      10.0%        Distributed Denial of Service flood
PortScan       100      10.0%        Reconnaissance and service enumeration probe
BruteForce     100      10.0%        Credential guessing / authentication attack
--------------------------------------------------------------------------------
TOTAL         1,000     100.0%

================================================================================
3. FEATURE SCHEMA & COLUMN SPECIFICATIONS
================================================================================

The 20 required network flow feature columns (in exact schema order):

 #   Column Name                  Type      Range / Values    Description
--------------------------------------------------------------------------------
 1.  duration                     Float     0.0 - 300.0       Connection duration in seconds
 2.  protocol_type                String    tcp, udp, icmp    Transport protocol utilized
 3.  src_bytes                    Float     0 - 50,000        Bytes transferred from source to destination
 4.  dst_bytes                    Float     0 - 100,000       Bytes transferred from destination to source
 5.  land                         Float     0 or 1            1 if source/dest IP & port are equal (spoofed)
 6.  wrong_fragment               Float     0, 1, 2           Number of erroneous/corrupted packet fragments
 7.  urgent                       Float     0 or 1            Count of packets with Urgent pointer flag
 8.  hot                          Float     0 - 5             Number of 'hot' indicators (e.g. system access)
 9.  num_failed_logins            Float     0 - 10            Count of failed authentication attempts
 10. logged_in                    Float     0 or 1            1 if successfully authenticated; 0 otherwise
 11. num_compromised              Float     0 - 5             Number of compromised machine conditions
 12. count                        Float     1 - 511           Connections to same host in past 2 seconds
 13. srv_count                    Float     1 - 511           Connections to same service in past 2 seconds
 14. same_srv_rate                Float     0.0 - 1.0         Ratio of connections to the same service
 15. diff_srv_rate                Float     0.0 - 1.0         Ratio of connections to different services
 16. dst_host_count               Float     1 - 255           Count of connections to destination host IP
 17. dst_host_srv_count           Float     1 - 255           Count of connections to destination port/srv
 18. dst_host_same_srv_rate       Float     0.0 - 1.0         Ratio of connections to same srv at dest host
 19. dst_host_diff_srv_rate       Float     0.0 - 1.0         Ratio of connections to diff srv at dest host
 20. dst_host_same_src_port_rate  Float     0.0 - 1.0         Ratio of connections from same source port

Label Column (Present in cyber_guardian_demo_labelled.csv only):
 21. label                        String    BENIGN, DoS, DDoS, PortScan, BruteForce

================================================================================
4. HOW TO USE IN THE STREAMLIT WEB APPLICATION
================================================================================

Step 1: Launch Cyber Guardian AI
    Run the command:
        streamlit run app.py
    Or access your hosted cloud deployment.

Step 2: Navigate to AI Threat Detection
    In the sidebar navigation menu, click "AI Threat Detection".

Step 3: Upload the CSV
    - In the "Batch Flow Telemetry Upload" section, drag and drop
      `cyber_guardian_demo.csv` or click "Browse files".
    - The application automatically validates the 20 required flow columns.
    - You will see a success notification: "Loaded 1,000 records successfully!"

Step 4: Execute Machine Learning Analysis
    - Click the prominent blue button: "⚡ Execute Machine Learning Analysis".
    - The Random Forest classification pipeline processes the 1,000 flows.
    - Review the generated metrics:
      * Total Records Evaluated: 1,000
      * Normal Traffic vs. High-Risk Threats identified
      * Multi-class distribution (BENIGN, DoS, DDoS, PortScan, BruteForce)
      * Enriched incident triage table with confidence scores, severity flags,
        and recommended defensive response playbooks.

================================================================================
5. VERIFICATION & VALIDATION CONFIRMATION
================================================================================

This dataset has been verified through automated testing directly with
the application's internal pipeline modules:
- src.data_processing.load_csv_bytes: PASSED (0 parsing errors)
- src.data_processing.validate_dataframe: PASSED (all 20 columns recognized)
- src.data_processing.preprocess_dataframe: PASSED (type coercion & scaling ready)
- src.detection.run_prediction: PASSED (1,000 / 1,000 records successfully classified)
================================================================================

import xml.etree.ElementTree as ET
import glob, csv, os, sys

output_csv = sys.argv[1] if len(sys.argv) > 1 else "/home/runner/_work/csv_results/bazel_tests_all_results.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Cache markers found in python test files
FILE_MARKERS_CACHE = {}

def get_file_markers(file_path):
    """Scan python test source file for @pytest.mark.<name> decorators."""
    if file_path in FILE_MARKERS_CACHE:
        return FILE_MARKERS_CACHE[file_path]
        
    markers = set()
    if "golden_config" in file_path:
        markers.add("golden_config")
    if "for_8_devices" in file_path:
        markers.add("for_8_devices")
    if "fp64" in file_path:
        markers.add("fp64")
    if "tpu" in file_path:
        markers.add("tpu")
    if "gpu" in file_path:
        markers.add("gpu")

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "@pytest.mark." in line:
                        mark = line.split("@pytest.mark.")[1].split("(")[0].split()[0].strip()
                        if mark and mark not in ["skip", "skipif", "parameterized"]:
                            markers.add(mark)
        except Exception:
            pass
            
    FILE_MARKERS_CACHE[file_path] = markers
    return markers

def determine_markers(file_path, name, raw_class):
    """Determine markers from file annotations and test/class names."""
    file_marks = set(get_file_markers(file_path))
    
    for keyword in ["golden_config", "for_8_devices", "fp64", "tpu", "gpu", "high_cpu", "gs_login"]:
        if keyword in name.lower() or keyword in raw_class.lower():
            file_marks.add(keyword)
            
    return " ".join(sorted(file_marks))

def parse_failure_content(raw_text):
    """Clean raw failure text to extract assertion errors while ignoring HTML/logging noise."""
    if not raw_text:
        return ""
        
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    clean_lines = [
        l for l in lines 
        if not l.startswith("<") and not l.startswith("+ ERROR") and "tpu_info.py" not in l and "importing.py" not in l
    ]
    if not clean_lines:
        clean_lines = lines

    key_lines = [
        l for l in clean_lines 
        if any(kw in l for kw in ["AssertionError", "Error:", "Exception:", "self.assert", "Diff", "Mismatch", "ImportError"])
    ]
    
    if key_lines:
        msg = " | ".join(key_lines[:3])
    else:
        msg = " ".join(clean_lines[:4])
        
    return msg

def extract_detailed_error(tc, root, xml_file):
    """Extract detailed failure traceback & exception message from testcase, system-out, or test.log"""
    failure = tc.find("failure")
    error = tc.find("error")
    elem = failure if failure is not None else error
    
    msg = ""
    if elem is not None:
        raw_content = elem.text or elem.attrib.get("message", "")
        msg = parse_failure_content(raw_content)
        
    if not msg or "exited with error code" in msg or len(msg) < 15:
        sys_out = tc.find("system-out") or root.find("system-out") or tc.find("system-err") or root.find("system-err")
        sys_text = ""
        if sys_out is not None and sys_out.text:
            sys_text = sys_out.text.strip()
            
        if not sys_text:
            log_path = os.path.join(os.path.dirname(xml_file), "test.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                        sys_text = lf.read().strip()
                except Exception:
                    pass
                    
        if sys_text:
            msg = parse_failure_content(sys_text)
                
    return " ".join(msg.splitlines())[:300]

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "module", "name", "file", "doc", "markers", "status", "message", "duration"])
    
    xml_files = glob.glob("bazel-testlogs/**/test.xml", recursive=True)
    print(f"Found {len(xml_files)} test.xml files.")
    
    for xml_file in xml_files:
        norm_path = os.path.normpath(xml_file)
        parts = norm_path.split(os.sep)
        
        if parts[0] == "bazel-testlogs":
            rel_parts = parts[1:-1]
        else:
            rel_parts = parts[:-1]
            
        rel_path = "/".join(rel_parts)
        file_path = f"{rel_path}.py"
        module_path = rel_path.replace("/", ".")
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for tc in root.iter("testcase"):
                name = tc.attrib.get("name", "")
                raw_class = tc.attrib.get("classname", "")
                time = tc.attrib.get("time", "0.0")
                
                clean_class = raw_class.replace("__main__.", "").replace("__main__", "")
                if clean_class:
                    test_id = f"{file_path}::{clean_class}::{name}"
                else:
                    test_id = f"{file_path}::{name}"
                
                markers = determine_markers(file_path, name, raw_class)
                
                status = "passed"
                msg = ""
                failure = tc.find("failure")
                error = tc.find("error")
                skipped = tc.find("skipped")
                
                if failure is not None or error is not None:
                    status = "failed"
                    msg = extract_detailed_error(tc, root, xml_file)
                elif skipped is not None:
                    status = "skipped"
                
                writer.writerow([test_id, module_path, name, file_path, "", markers, status, msg, time])
        except Exception as e:
            pass

print(f"Successfully generated {output_csv}!")

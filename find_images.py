#!python3
import pymupdf
import re
import os
import argparse
import json


ISSUES_TRESHOLD = 90
LABEL_DIST = 20

# Global flags
debug=False
save_figures = False
save_tables = False
output_dir = os.getcwd()


def near_any(box1:pymupdf.Rect, box2:pymupdf.Rect, treshold = LABEL_DIST):
    dx = min(abs(box1.x0-box2.x0),
             abs(box1.x0-box2.x1),
             abs(box1.x1-box2.x1),
             abs(box1.x1-box2.x0))
    dy = min(abs(box1.y0-box2.y0),
             abs(box1.y0-box2.y1),
             abs(box1.y1-box2.y1),
             abs(box1.y1-box2.y0))
    return dx < treshold and dy<treshold

def near(box1:pymupdf.Rect, box2:pymupdf.Rect, treshold = LABEL_DIST, dir="any"):
    """
    checks if two box2 is near box1 in a given direction
    E.g. for `dir="above"` checks if `box2` is above `box1`
    """
    if dir=="above": # box2 above box1
        return 0 < (box1.y0-box2.y1) < treshold
    if dir=="below": # box2 below box1
        return 0 < (box2.y0-box1.y1) < treshold \
              or 0 < (box2.y0-box1.y0) < treshold
    if dir=="left": # box2 left box1
        return 0 < (box1.x0-box2.x1) < treshold
    if dir=="right": # box2 right box1
        return 0 < (box2.x0-box1.x1) < treshold
    if dir=="any":
        return near_any(box1,box2,treshold)
    raise Exception("Invalid direction: "+dir)


def inside(box1:pymupdf.Rect, box2:pymupdf.Rect|list|None, tolerance = 1):
    """
    checks if `box1` is inside `box2` with a given `tolerance`
    """
    if box2 is None: return False
    if isinstance(box2,list):
        return all( inside(box1,b) for b in box2)
    else:
        return box1.x0-box2.x0 > -1 and box2.x1-box1.x1>-1 and \
               box1.y0>=box2.y0 and box1.y1<=box2.y1

def cumulative_bbox(start_bbox: pymupdf.Rect, boxes):
    """
    Combines multiple bounding `boxes` starting
    from an initial `start_bbox`, corresponding to the caption
    """
    if debug: print("\tCombining",len(boxes),"bboxes ->",end="")
    bbox = start_bbox
    for p in boxes:
        r = p['rect']
        if r.x0<bbox.x0:
            bbox.x0 = r.x0
        if r.y0<bbox.y0:
            bbox.y0 = r.y0
        if r.x1>bbox.x1:
            bbox.x1 = r.x1
        if r.y1>bbox.y1:
            bbox.y1 = r.y1
    if debug: print(bbox)
    return bbox

def tuple_to_rect(box):
    return pymupdf.Rect(x0=box[0],y0=box[1],x1=box[2],y1=box[3])

def detect_figure(caption, paths, images):
    br = tuple_to_rect(caption)
    horizontal = (br.x1-br.x0) > (br.y1-br.y0)
    if horizontal:
        linked_paths = [ p for p in paths if near(br,p['rect'],2*LABEL_DIST,dir="above") and 
                     p['rect'].y0<=br.y0 and p['rect'].x0>=br.x0]
    else:
        linked_paths = [ p for p in paths if near(br,p['rect'],dir="left") and 
                     p['rect'].y1<=br.y1 and p['rect'].x0<=br.x0]
    if debug: print("\tCaption",br)
    if debug: print("\tFound",len(linked_paths),"close paths out of ",len(paths))

    found = len(linked_paths)>0

    paths = [ p for p in paths if p not in linked_paths]
    while found:
        found = False
        for p in paths:
            rp = p['rect']
            #if p not in linked_paths:
            for lp in linked_paths:
                rl = lp['rect']
                if near(rp,rl,2*LABEL_DIST) and rp.y0<=br.y0 and rp.x0>=br.x0:
                    linked_paths.append(p)
                    found = True
                    break
        paths = [ p for p in paths if p not in linked_paths]

    if len(linked_paths)==0:
        if debug: print("\tLooking into ",len(images),"images")
        for i in images:
            ib = tuple_to_rect(i['bbox'])
            if (horizontal and near(br,ib,2*LABEL_DIST,dir="above")) or \
               (not horizontal and near(br,ib,2*LABEL_DIST,dir="left")):
                linked_paths.append({'rect':ib})
                if debug: print("\tFound close image")
                found = True
        if not found:
            return None
    
    if debug: print("\tCombining",len(linked_paths),"bboxes")
    return cumulative_bbox(br, linked_paths)


def parallel(path1, path2, horizontal: bool, tolerance=1.5):
    p1 = path1['rect']
    p2 = path2['rect']
    if horizontal:
        return abs(p1.y0-p1.y1)<tolerance and abs(p2.y0-p2.y1)<tolerance and \
               abs(p1.x0-p2.x0)<tolerance and abs(p1.x1-p2.x1)<tolerance
    else:
        return (abs(p1.x0-p1.x1)<tolerance and abs(p2.x0-p2.x1)<tolerance and \
            abs(p1.y0-p2.y0)<tolerance and abs(p1.y1-p2.y1)<tolerance)


def detect_table(caption, paths, exclude=None):
    br = tuple_to_rect(caption)
    horizontal = (br.x1-br.x0) > (br.y1-br.y0)
    if horizontal:
        linked_paths = [ p for p in paths if near(br,p['rect'],2*LABEL_DIST,dir="below") ]
    else:
        linked_paths = [ p for p in paths if near(br,p['rect'],dir="right") and 
                            p['rect'].y1<=br.y1 and p['rect'].x0>=br.x0]
    if debug: print("\tCaption",br, "Horizontal: ", horizontal)
    if debug: print("\tFound",len(linked_paths),"close paths out of ",len(paths))
    if debug: 
        for p in linked_paths: print("\t\t",p['rect'])

    found = len(linked_paths)>0

    while found:
        found = False
        for p in paths:
            rp = p['rect']
            if p not in linked_paths:
                for lp in linked_paths:
                    rl = lp['rect']
                    if near(rp,rl) and rp.y0>=br.y0 and rp.x0>=br.x0:
                        linked_paths.append(p)
                        found = True
                        break
    
    rules = []
    for p in paths:
        rp = p['rect']
        for lp in linked_paths:
            if p not in linked_paths \
               and parallel(p,lp,horizontal) \
               and ((horizontal and rp.y0>=lp['rect'].y0) or \
                    ((not horizontal) and rp.x0>=lp['rect'].x0)) \
               and not inside(rp,exclude):
                if debug: print("\t\tR: ",rp)
                rules.append(p)
                break
    if debug and len(rules)>0: print("\tFound ",len(rules),"rules")
    linked_paths.extend(rules)

    if len(linked_paths)==0:
        return None
    
    return cumulative_bbox(br, linked_paths)

# Patterns
fig_pattern = r"^[fF]ig(?:ure)?[.:-]?\s*(\d+)"
tbl_pattern = r"^[tT]able[^ \t]*\s*(\d+|[IVX]+)"
doi_pattern = r"(10[.]\d{4,}(?:[.]\d+)*/(?:(?!['\"&])\S)+)"

def extract_paper_info(path, doc, start=0, end=-1):
    global save_tables
    global save_figures

    if debug: print("Save tables:",save_tables)

    fig_prefix = os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0])

    matches = [re.search(doi_pattern, b[4]) for b in doc[start].get_text("blocks")]
    if len([m for m in matches if m is not None]) == 0:
        if debug: print("Looking for doi on page 0")
        matches = [re.search(doi_pattern, b[4]) for b in doc[0].get_text("blocks")]
    matches = [m.group(1) for m in matches if m is not None]
    doi = max(matches, key=len)

    num_pages = len(doc)
    if end==-1:
        end = num_pages
    else:
        num_pages = end-start
    num_figures = 0
    num_tables = 0

    figures = []
    tables = []
    # Extract images from each page
    for page_num in range(start, end):
        page = doc[page_num]
        print(f"{fig_prefix} - {page_num-start}/{num_pages}", end="\r")
        blocks = page.get_text("blocks")
        paths = page.get_drawings()
        images = page.get_image_info()
        tbl_candidates = []
        for b in blocks:
            block_text = b[4]
            match = re.search(fig_pattern, block_text)
            if match is not None:
                fig_num = int(match.group(1))
                num_figures = max(num_figures, fig_num)
                if debug: print("Found figure",fig_num,"on page",page_num+1)
                path_bbox = detect_figure(b,paths,images)
                if path_bbox is not None:
                    if save_figures:
                        shot = page.get_pixmap(clip=path_bbox, dpi=300)
                        shot.save(f"{fig_prefix}_p{page_num+1}_fig{fig_num}.png")
                    figures.append( {"figure":fig_num, "page":page_num+1} )
            
            match = re.search(tbl_pattern, block_text)
            if match is not None:
                tbl_num = int(match.group(1))
                num_tables = max(num_tables, tbl_num)
                if debug: print("Found table",tbl_num,"on page",page_num+1)
                path_bbox = detect_table(b,paths)
                if path_bbox is not None:
                    tbl_candidates.append((fig_prefix,page_num,tbl_num,b,path_bbox))
        
        for t1 in tbl_candidates:
            bbox = t1[4]
            contained = []
            for t2 in tbl_candidates:
                if t1!=t2 and inside(t2[4],t1[4]):
                    if debug: print("Table ",t2[2],"inside",t1[2])
                    contained.append(t2[4])
            if len(contained)>0:
                bbox = detect_table(t1[3], paths, contained)
            if save_tables:
                shot = page.get_pixmap(clip=bbox, dpi=300)
                shot.save(f"{t1[0]}_p{t1[1]+1}_tbl{t1[2]}.png")
            tables.append( {"table":t1[2], "page":t1[1]+1} )

    paper_info = {
        "path":path,
        "doi":doi,
        "num_pages": num_pages,
        "num_figures": num_figures,
        "figures": figures,
        "num_tables": num_tables,
        "tables": tables
    }
    if num_figures>len(figures):
        missing_figures = [nf for nf in range(1,num_figures) if nf not in [nf['figure'] for nf in figures] ]
        print("Warning: missing figures in",path, missing_figures)
    if num_tables>len(tables):
        missing_tables = [nt for nt in range(1,num_tables) if nt not in [nt['table'] for nt in tables] ]
        print("Warning: missing tables in",path,missing_tables)
    return [paper_info]


doi_regex_issue = r"\bno. (10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!['\"&])\S)+)\b" # works for IEEE TSE

def process_pdf(path, doc):
    num_pages = len(doc)
    if num_pages>ISSUES_TRESHOLD:
        if debug: print("Extracting papers from issue...")
        matches=[ re.search(doi_regex_issue,page.get_text()) for page in doc]
        dois = [ (m and m.group(1),p) for p,m in enumerate(matches) if m is not None]
        num_papers = len(dois)
        print(f" 0/{num_papers}                           ", end="\r")
        papers = []
        for i,m in enumerate(dois):
            start_p = m[1]
            if i==len(dois)-1:
                end_p=num_pages
            else:
                end_p=dois[i+1][1]
            if debug: print("Paper",i,":",start_p,"-",end_p)
            paper_info = extract_paper_info(path, doc, start_p, end_p)[0]
            papers.append(paper_info)
            print(f" {i+1}/{num_papers}", end="\r")
        print()
        return papers
    else:
        return extract_paper_info(path,doc)



def analyze_pdf(path, start_page=0, end_page=-1):
    if os.path.isdir(path):
        files = [os.path.join(path,x) for x in os.listdir(path) if x.endswith(".pdf")]
    else:
        files = [path]

    n = len(files)
    print(f"Processing {n} files")
    count = 0
    result = []
    for p in files:
        print(f"{count}/{n}",end="\r")
        # try:
        doc = pymupdf.open(p)
        result.extend(extract_paper_info(p, doc, start_page, end_page))
        # except Exception as e:
        #     result.append({"path":p,"doi":"NONE"})
        #     print("Error while processing file:",p,":",e)
        count += 1
    print(f"{count}/{n}")
    #result = [extract_paper_info(p) for p in files]
    return result



def main():
    global debug
    global save_tables
    global save_figures
    global output_dir

    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    # Define expected arguments
    parser.add_argument("path", help="A folder or a single .pdf file")
    parser.add_argument("--output","-o", type=str, help="Output file or folder (if folder, uses 'papers.json' as filename)", default="papers.json")
    parser.add_argument("--figures", "-f", action="store_true", help="Save figures snapshots")
    parser.add_argument("--tables", "-t", action="store_true", help="Save tables snapshots")
    parser.add_argument("--debug","-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--start","-s", type=int, help="Start page", default=0)
    parser.add_argument("--end","-e", type=int, help="End page", default=-1)
    
    # Parse arguments
    args = parser.parse_args()

    debug = args.debug
    save_figures = args.figures
    save_tables = args.tables

    # Compute output_dir before processing PDFs
    output_path = args.output
    if output_path is None or output_path == "-":
        output_dir = os.getcwd()
    else:
        if output_path.endswith(os.sep) or (os.altsep and output_path.endswith(os.altsep)) or os.path.isdir(output_path):
            output_dir = output_path
        else:
            output_dir = os.path.dirname(output_path) or os.getcwd()

    res = analyze_pdf(args.path, args.start-1, args.end)
    restxt = json.dumps(res)

    if output_path is None or output_path == "-":
        print(restxt)
    else:
        output_file = os.path.join(output_dir, "papers.json") if os.path.isdir(output_path) else output_path
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as fp:
            fp.write(restxt)

if __name__ == "__main__":
    main()
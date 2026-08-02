"""
DCL Crypto Suite — detectors for the 6 crypto-specialized MCP tools:
dcl_commit, dcl_evaluate_mev, dcl_evaluate_jailbreak_crypto,
dcl_evaluate_signal, dcl_evaluate_trade, dcl_evaluate_wallet.

Mirrors the style of dcl_core.py's detect_secrets/detect_pii: pure,
stateless regex/heuristic scanners that return a plain dict. Chain
commitment, hashing, and MCP/pydantic wiring stay in mcp_server.py,
exactly like the existing secrets/PII tools.

Category codes below (W1-W4, M1-M4, X1-X4, V1-V4) follow the free
instruction-only checklists published alongside the live tools in the
dcl-skills repo (skills/crypto/*/references/free-checklist.md) — the
live paid tools apply the same detection logic server-side instead of
asking the calling agent to do it by hand.

Note on verdict collapsing: the free checklists for MEV Compliance and
Prompt Firewall Crypto define a three-state local verdict (COMMIT / WARN /
NO_COMMIT) for a single "major" finding. The live MCP tools' documented
output schema only exposes two verdict states (COMMIT | NO_COMMIT), so a
lone WARN-tier finding is collapsed into NO_COMMIT here, but with a
higher (less severe) confidence score and a `reason` that says so
explicitly, so downstream callers can still tell a single soft flag
apart from a hard multi-finding block if they inspect confidence/reason.
"""
import re
from typing import List, Optional, Tuple


def _redact(s: str, keep_start: int = 2, keep_end: int = 4) -> str:
    """Mask a matched string, keeping only a few edge chars. Mirrors dcl_core._redact."""
    if len(s) <= keep_start + keep_end:
        return "*" * len(s)
    middle = "*" * max(4, len(s) - keep_start - keep_end)
    return f"{s[:keep_start]}{middle}{s[-keep_end:]}"


# ════════════════════════════════════════════════════════════════════════════════
# BIP-39 English wordlist (official, from bitcoin/bips) — used only for the
# seed-phrase check (W1). Function words ("the", "a", "is", "of", ...) are not
# part of this list, which keeps false positives on ordinary prose low: an
# unrelated 12-24 word run of plain English is very unlikely to consist
# entirely of BIP-39 words.
# ════════════════════════════════════════════════════════════════════════════════
BIP39_WORDS = frozenset({
    'abandon','ability','able','about','above','absent','absorb','abstract','absurd','abuse',
    'access','accident','account','accuse','achieve','acid','acoustic','acquire','across','act',
    'action','actor','actress','actual','adapt','add','addict','address','adjust','admit','adult',
    'advance','advice','aerobic','affair','afford','afraid','again','age','agent','agree','ahead',
    'aim','air','airport','aisle','alarm','album','alcohol','alert','alien','all','alley','allow',
    'almost','alone','alpha','already','also','alter','always','amateur','amazing','among','amount',
    'amused','analyst','anchor','ancient','anger','angle','angry','animal','ankle','announce',
    'annual','another','answer','antenna','antique','anxiety','any','apart','apology','appear',
    'apple','approve','april','arch','arctic','area','arena','argue','arm','armed','armor','army',
    'around','arrange','arrest','arrive','arrow','art','artefact','artist','artwork','ask','aspect',
    'assault','asset','assist','assume','asthma','athlete','atom','attack','attend','attitude',
    'attract','auction','audit','august','aunt','author','auto','autumn','average','avocado',
    'avoid','awake','aware','away','awesome','awful','awkward','axis','baby','bachelor','bacon',
    'badge','bag','balance','balcony','ball','bamboo','banana','banner','bar','barely','bargain',
    'barrel','base','basic','basket','battle','beach','bean','beauty','because','become','beef',
    'before','begin','behave','behind','believe','below','belt','bench','benefit','best','betray',
    'better','between','beyond','bicycle','bid','bike','bind','biology','bird','birth','bitter',
    'black','blade','blame','blanket','blast','bleak','bless','blind','blood','blossom','blouse',
    'blue','blur','blush','board','boat','body','boil','bomb','bone','bonus','book','boost',
    'border','boring','borrow','boss','bottom','bounce','box','boy','bracket','brain','brand',
    'brass','brave','bread','breeze','brick','bridge','brief','bright','bring','brisk','broccoli',
    'broken','bronze','broom','brother','brown','brush','bubble','buddy','budget','buffalo','build',
    'bulb','bulk','bullet','bundle','bunker','burden','burger','burst','bus','business','busy',
    'butter','buyer','buzz','cabbage','cabin','cable','cactus','cage','cake','call','calm','camera',
    'camp','can','canal','cancel','candy','cannon','canoe','canvas','canyon','capable','capital',
    'captain','car','carbon','card','cargo','carpet','carry','cart','case','cash','casino','castle',
    'casual','cat','catalog','catch','category','cattle','caught','cause','caution','cave',
    'ceiling','celery','cement','census','century','cereal','certain','chair','chalk','champion',
    'change','chaos','chapter','charge','chase','chat','cheap','check','cheese','chef','cherry',
    'chest','chicken','chief','child','chimney','choice','choose','chronic','chuckle','chunk',
    'churn','cigar','cinnamon','circle','citizen','city','civil','claim','clap','clarify','claw',
    'clay','clean','clerk','clever','click','client','cliff','climb','clinic','clip','clock','clog',
    'close','cloth','cloud','clown','club','clump','cluster','clutch','coach','coast','coconut',
    'code','coffee','coil','coin','collect','color','column','combine','come','comfort','comic',
    'common','company','concert','conduct','confirm','congress','connect','consider','control',
    'convince','cook','cool','copper','copy','coral','core','corn','correct','cost','cotton',
    'couch','country','couple','course','cousin','cover','coyote','crack','cradle','craft','cram',
    'crane','crash','crater','crawl','crazy','cream','credit','creek','crew','cricket','crime',
    'crisp','critic','crop','cross','crouch','crowd','crucial','cruel','cruise','crumble','crunch',
    'crush','cry','crystal','cube','culture','cup','cupboard','curious','current','curtain','curve',
    'cushion','custom','cute','cycle','dad','damage','damp','dance','danger','daring','dash',
    'daughter','dawn','day','deal','debate','debris','decade','december','decide','decline',
    'decorate','decrease','deer','defense','define','defy','degree','delay','deliver','demand',
    'demise','denial','dentist','deny','depart','depend','deposit','depth','deputy','derive',
    'describe','desert','design','desk','despair','destroy','detail','detect','develop','device',
    'devote','diagram','dial','diamond','diary','dice','diesel','diet','differ','digital','dignity',
    'dilemma','dinner','dinosaur','direct','dirt','disagree','discover','disease','dish','dismiss',
    'disorder','display','distance','divert','divide','divorce','dizzy','doctor','document','dog',
    'doll','dolphin','domain','donate','donkey','donor','door','dose','double','dove','draft',
    'dragon','drama','drastic','draw','dream','dress','drift','drill','drink','drip','drive','drop',
    'drum','dry','duck','dumb','dune','during','dust','dutch','duty','dwarf','dynamic','eager',
    'eagle','early','earn','earth','easily','east','easy','echo','ecology','economy','edge','edit',
    'educate','effort','egg','eight','either','elbow','elder','electric','elegant','element',
    'elephant','elevator','elite','else','embark','embody','embrace','emerge','emotion','employ',
    'empower','empty','enable','enact','end','endless','endorse','enemy','energy','enforce',
    'engage','engine','enhance','enjoy','enlist','enough','enrich','enroll','ensure','enter',
    'entire','entry','envelope','episode','equal','equip','era','erase','erode','erosion','error',
    'erupt','escape','essay','essence','estate','eternal','ethics','evidence','evil','evoke',
    'evolve','exact','example','excess','exchange','excite','exclude','excuse','execute','exercise',
    'exhaust','exhibit','exile','exist','exit','exotic','expand','expect','expire','explain',
    'expose','express','extend','extra','eye','eyebrow','fabric','face','faculty','fade','faint',
    'faith','fall','false','fame','family','famous','fan','fancy','fantasy','farm','fashion','fat',
    'fatal','father','fatigue','fault','favorite','feature','february','federal','fee','feed',
    'feel','female','fence','festival','fetch','fever','few','fiber','fiction','field','figure',
    'file','film','filter','final','find','fine','finger','finish','fire','firm','first','fiscal',
    'fish','fit','fitness','fix','flag','flame','flash','flat','flavor','flee','flight','flip',
    'float','flock','floor','flower','fluid','flush','fly','foam','focus','fog','foil','fold',
    'follow','food','foot','force','forest','forget','fork','fortune','forum','forward','fossil',
    'foster','found','fox','fragile','frame','frequent','fresh','friend','fringe','frog','front',
    'frost','frown','frozen','fruit','fuel','fun','funny','furnace','fury','future','gadget','gain',
    'galaxy','gallery','game','gap','garage','garbage','garden','garlic','garment','gas','gasp',
    'gate','gather','gauge','gaze','general','genius','genre','gentle','genuine','gesture','ghost',
    'giant','gift','giggle','ginger','giraffe','girl','give','glad','glance','glare','glass',
    'glide','glimpse','globe','gloom','glory','glove','glow','glue','goat','goddess','gold','good',
    'goose','gorilla','gospel','gossip','govern','gown','grab','grace','grain','grant','grape',
    'grass','gravity','great','green','grid','grief','grit','grocery','group','grow','grunt',
    'guard','guess','guide','guilt','guitar','gun','gym','habit','hair','half','hammer','hamster',
    'hand','happy','harbor','hard','harsh','harvest','hat','have','hawk','hazard','head','health',
    'heart','heavy','hedgehog','height','hello','helmet','help','hen','hero','hidden','high','hill',
    'hint','hip','hire','history','hobby','hockey','hold','hole','holiday','hollow','home','honey',
    'hood','hope','horn','horror','horse','hospital','host','hotel','hour','hover','hub','huge',
    'human','humble','humor','hundred','hungry','hunt','hurdle','hurry','hurt','husband','hybrid',
    'ice','icon','idea','identify','idle','ignore','ill','illegal','illness','image','imitate',
    'immense','immune','impact','impose','improve','impulse','inch','include','income','increase',
    'index','indicate','indoor','industry','infant','inflict','inform','inhale','inherit','initial',
    'inject','injury','inmate','inner','innocent','input','inquiry','insane','insect','inside',
    'inspire','install','intact','interest','into','invest','invite','involve','iron','island',
    'isolate','issue','item','ivory','jacket','jaguar','jar','jazz','jealous','jeans','jelly',
    'jewel','job','join','joke','journey','joy','judge','juice','jump','jungle','junior','junk',
    'just','kangaroo','keen','keep','ketchup','key','kick','kid','kidney','kind','kingdom','kiss',
    'kit','kitchen','kite','kitten','kiwi','knee','knife','knock','know','lab','label','labor',
    'ladder','lady','lake','lamp','language','laptop','large','later','latin','laugh','laundry',
    'lava','law','lawn','lawsuit','layer','lazy','leader','leaf','learn','leave','lecture','left',
    'leg','legal','legend','leisure','lemon','lend','length','lens','leopard','lesson','letter',
    'level','liar','liberty','library','license','life','lift','light','like','limb','limit','link',
    'lion','liquid','list','little','live','lizard','load','loan','lobster','local','lock','logic',
    'lonely','long','loop','lottery','loud','lounge','love','loyal','lucky','luggage','lumber',
    'lunar','lunch','luxury','lyrics','machine','mad','magic','magnet','maid','mail','main','major',
    'make','mammal','man','manage','mandate','mango','mansion','manual','maple','marble','march',
    'margin','marine','market','marriage','mask','mass','master','match','material','math','matrix',
    'matter','maximum','maze','meadow','mean','measure','meat','mechanic','medal','media','melody',
    'melt','member','memory','mention','menu','mercy','merge','merit','merry','mesh','message',
    'metal','method','middle','midnight','milk','million','mimic','mind','minimum','minor','minute',
    'miracle','mirror','misery','miss','mistake','mix','mixed','mixture','mobile','model','modify',
    'mom','moment','monitor','monkey','monster','month','moon','moral','more','morning','mosquito',
    'mother','motion','motor','mountain','mouse','move','movie','much','muffin','mule','multiply',
    'muscle','museum','mushroom','music','must','mutual','myself','mystery','myth','naive','name',
    'napkin','narrow','nasty','nation','nature','near','neck','need','negative','neglect','neither',
    'nephew','nerve','nest','net','network','neutral','never','news','next','nice','night','noble',
    'noise','nominee','noodle','normal','north','nose','notable','note','nothing','notice','novel',
    'now','nuclear','number','nurse','nut','oak','obey','object','oblige','obscure','observe',
    'obtain','obvious','occur','ocean','october','odor','off','offer','office','often','oil','okay',
    'old','olive','olympic','omit','once','one','onion','online','only','open','opera','opinion',
    'oppose','option','orange','orbit','orchard','order','ordinary','organ','orient','original',
    'orphan','ostrich','other','outdoor','outer','output','outside','oval','oven','over','own',
    'owner','oxygen','oyster','ozone','pact','paddle','page','pair','palace','palm','panda','panel',
    'panic','panther','paper','parade','parent','park','parrot','party','pass','patch','path',
    'patient','patrol','pattern','pause','pave','payment','peace','peanut','pear','peasant',
    'pelican','pen','penalty','pencil','people','pepper','perfect','permit','person','pet','phone',
    'photo','phrase','physical','piano','picnic','picture','piece','pig','pigeon','pill','pilot',
    'pink','pioneer','pipe','pistol','pitch','pizza','place','planet','plastic','plate','play',
    'please','pledge','pluck','plug','plunge','poem','poet','point','polar','pole','police','pond',
    'pony','pool','popular','portion','position','possible','post','potato','pottery','poverty',
    'powder','power','practice','praise','predict','prefer','prepare','present','pretty','prevent',
    'price','pride','primary','print','priority','prison','private','prize','problem','process',
    'produce','profit','program','project','promote','proof','property','prosper','protect','proud',
    'provide','public','pudding','pull','pulp','pulse','pumpkin','punch','pupil','puppy','purchase',
    'purity','purpose','purse','push','put','puzzle','pyramid','quality','quantum','quarter',
    'question','quick','quit','quiz','quote','rabbit','raccoon','race','rack','radar','radio',
    'rail','rain','raise','rally','ramp','ranch','random','range','rapid','rare','rate','rather',
    'raven','raw','razor','ready','real','reason','rebel','rebuild','recall','receive','recipe',
    'record','recycle','reduce','reflect','reform','refuse','region','regret','regular','reject',
    'relax','release','relief','rely','remain','remember','remind','remove','render','renew','rent',
    'reopen','repair','repeat','replace','report','require','rescue','resemble','resist','resource',
    'response','result','retire','retreat','return','reunion','reveal','review','reward','rhythm',
    'rib','ribbon','rice','rich','ride','ridge','rifle','right','rigid','ring','riot','ripple',
    'risk','ritual','rival','river','road','roast','robot','robust','rocket','romance','roof',
    'rookie','room','rose','rotate','rough','round','route','royal','rubber','rude','rug','rule',
    'run','runway','rural','sad','saddle','sadness','safe','sail','salad','salmon','salon','salt',
    'salute','same','sample','sand','satisfy','satoshi','sauce','sausage','save','say','scale',
    'scan','scare','scatter','scene','scheme','school','science','scissors','scorpion','scout',
    'scrap','screen','script','scrub','sea','search','season','seat','second','secret','section',
    'security','seed','seek','segment','select','sell','seminar','senior','sense','sentence',
    'series','service','session','settle','setup','seven','shadow','shaft','shallow','share','shed',
    'shell','sheriff','shield','shift','shine','ship','shiver','shock','shoe','shoot','shop',
    'short','shoulder','shove','shrimp','shrug','shuffle','shy','sibling','sick','side','siege',
    'sight','sign','silent','silk','silly','silver','similar','simple','since','sing','siren',
    'sister','situate','six','size','skate','sketch','ski','skill','skin','skirt','skull','slab',
    'slam','sleep','slender','slice','slide','slight','slim','slogan','slot','slow','slush','small',
    'smart','smile','smoke','smooth','snack','snake','snap','sniff','snow','soap','soccer','social',
    'sock','soda','soft','solar','soldier','solid','solution','solve','someone','song','soon',
    'sorry','sort','soul','sound','soup','source','south','space','spare','spatial','spawn','speak',
    'special','speed','spell','spend','sphere','spice','spider','spike','spin','spirit','split',
    'spoil','sponsor','spoon','sport','spot','spray','spread','spring','spy','square','squeeze',
    'squirrel','stable','stadium','staff','stage','stairs','stamp','stand','start','state','stay',
    'steak','steel','stem','step','stereo','stick','still','sting','stock','stomach','stone',
    'stool','story','stove','strategy','street','strike','strong','struggle','student','stuff',
    'stumble','style','subject','submit','subway','success','such','sudden','suffer','sugar',
    'suggest','suit','summer','sun','sunny','sunset','super','supply','supreme','sure','surface',
    'surge','surprise','surround','survey','suspect','sustain','swallow','swamp','swap','swarm',
    'swear','sweet','swift','swim','swing','switch','sword','symbol','symptom','syrup','system',
    'table','tackle','tag','tail','talent','talk','tank','tape','target','task','taste','tattoo',
    'taxi','teach','team','tell','ten','tenant','tennis','tent','term','test','text','thank','that',
    'theme','then','theory','there','they','thing','this','thought','three','thrive','throw',
    'thumb','thunder','ticket','tide','tiger','tilt','timber','time','tiny','tip','tired','tissue',
    'title','toast','tobacco','today','toddler','toe','together','toilet','token','tomato',
    'tomorrow','tone','tongue','tonight','tool','tooth','top','topic','topple','torch','tornado',
    'tortoise','toss','total','tourist','toward','tower','town','toy','track','trade','traffic',
    'tragic','train','transfer','trap','trash','travel','tray','treat','tree','trend','trial',
    'tribe','trick','trigger','trim','trip','trophy','trouble','truck','true','truly','trumpet',
    'trust','truth','try','tube','tuition','tumble','tuna','tunnel','turkey','turn','turtle',
    'twelve','twenty','twice','twin','twist','two','type','typical','ugly','umbrella','unable',
    'unaware','uncle','uncover','under','undo','unfair','unfold','unhappy','uniform','unique',
    'unit','universe','unknown','unlock','until','unusual','unveil','update','upgrade','uphold',
    'upon','upper','upset','urban','urge','usage','use','used','useful','useless','usual','utility',
    'vacant','vacuum','vague','valid','valley','valve','van','vanish','vapor','various','vast',
    'vault','vehicle','velvet','vendor','venture','venue','verb','verify','version','very','vessel',
    'veteran','viable','vibrant','vicious','victory','video','view','village','vintage','violin',
    'virtual','virus','visa','visit','visual','vital','vivid','vocal','voice','void','volcano',
    'volume','vote','voyage','wage','wagon','wait','walk','wall','walnut','want','warfare','warm',
    'warrior','wash','wasp','waste','water','wave','way','wealth','weapon','wear','weasel',
    'weather','web','wedding','weekend','weird','welcome','west','wet','whale','what','wheat',
    'wheel','when','where','whip','whisper','wide','width','wife','wild','will','win','window',
    'wine','wing','wink','winner','winter','wire','wisdom','wise','wish','witness','wolf','woman',
    'wonder','wood','wool','word','work','world','worry','worth','wrap','wreck','wrestle','wrist',
    'write','wrong','yard','year','yellow','you','young','youth','zebra','zero','zone','zoo',
})

# ════════════════════════════════════════════════════════════════════════════════
# W1-W4 — Wallet Guardian (dcl_evaluate_wallet)
# ════════════════════════════════════════════════════════════════════════════════
_PRIVATE_KEY_PATTERNS = [
    re.compile(r"\b(?:0x)?[0-9a-fA-F]{64}\b"),                        # raw hex private key
    re.compile(r"\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b"),               # WIF format
]

_WALLET_ADDRESS_PATTERNS = [
    re.compile(r"\b0x[a-fA-F0-9]{40}\b"),                              # Ethereum
    re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{24,33}\b"),                 # Bitcoin base58
    re.compile(r"\bbc1[a-z0-9]{25,39}\b"),                              # Bitcoin bech32
]

_WALLET_CONTEXT_WORDS = re.compile(
    r"(?i)\b(?:wallet|custody|custodial|sign(?:ing|ature)?|balance|withdraw(?:al)?)\b"
)
_API_KEY_OR_TOKEN = re.compile(
    r"(?i)(?:\b(?:api[_-]?key|access[_-]?token)\b\s*[:=]?\s*[A-Za-z0-9\-_.]{12,}"
    r"|\bbearer\s+[A-Za-z0-9\-_.=]{16,})"
)

_SEED_PHRASE_SIZES = (24, 12)


def _find_seed_phrases(text: str) -> List[dict]:
    """W1 — sequence of 12 or 24 BIP-39 words in a row, space-separated."""
    findings = []
    tokens = [(m.group(0), m.start()) for m in re.finditer(r"[a-z]+", text)]
    n = len(tokens)
    i = 0
    while i < n:
        matched = False
        for size in _SEED_PHRASE_SIZES:
            if i + size <= n:
                window = tokens[i:i + size]
                if all(w in BIP39_WORDS for w, _ in window):
                    start_pos = window[0][1]
                    end_pos = window[-1][1] + len(window[-1][0])
                    sample_words = " ".join(w for w, _ in window[:3])
                    findings.append({
                        "type": "seed_phrase",
                        "position": start_pos,
                        "end_position": end_pos,
                        "redacted_sample": f"{sample_words} ... ({size} words)",
                        "severity": "critical",
                    })
                    i += size
                    matched = True
                    break
        if not matched:
            i += 1
    return findings


def _find_private_keys(text: str) -> List[dict]:
    """W2 — 64-char hex (optionally 0x-prefixed) or WIF-format base58 key."""
    findings = []
    for pattern in _PRIVATE_KEY_PATTERNS:
        for m in pattern.finditer(text):
            findings.append({
                "type": "private_key",
                "position": m.start(),
                "end_position": m.end(),
                "redacted_sample": _redact(m.group(0)),
                "severity": "critical",
            })
    return findings


def _find_wallet_addresses(text: str) -> List[dict]:
    """W3 — Ethereum hex, Bitcoin base58, or bech32 address."""
    findings = []
    for pattern in _WALLET_ADDRESS_PATTERNS:
        for m in pattern.finditer(text):
            findings.append({
                "type": "wallet_address",
                "position": m.start(),
                "end_position": m.end(),
                "redacted_sample": _redact(m.group(0)),
                "severity": "major",
            })
    return findings


def _find_wallet_api_credentials(text: str) -> List[dict]:
    """W4 — API key/bearer token appearing near wallet/custody/signing terminology."""
    findings = []
    for m in _API_KEY_OR_TOKEN.finditer(text):
        window_start = max(0, m.start() - 60)
        window_end = min(len(text), m.end() + 60)
        if _WALLET_CONTEXT_WORDS.search(text[window_start:window_end]):
            findings.append({
                "type": "wallet_api_credential",
                "position": m.start(),
                "end_position": m.end(),
                "redacted_sample": _redact(m.group(0)),
                "severity": "major",
            })
    return findings


def _dedupe_by_span(findings: List[dict]) -> List[dict]:
    """Drop findings whose match span is fully contained in an earlier, already-kept one
    (e.g. a private-key hex run also matching inside a longer seed-phrase-shaped window)."""
    kept = []
    for f in sorted(findings, key=lambda f: (f["position"], -(f.get("end_position", f["position"])))):
        span = (f["position"], f.get("end_position", f["position"]))
        if any(span[0] >= k[0] and span[1] <= k[1] for k in [(k["position"], k.get("end_position", k["position"])) for k in kept]):
            continue
        kept.append(f)
    return kept


def detect_wallet(text: str) -> dict:
    """Scan for seed phrases, private keys, wallet addresses, and wallet-context
    API credentials (W1-W4). Any finding results in NO_COMMIT — wallet secrets
    have no safe threshold."""
    findings = _dedupe_by_span(
        _find_seed_phrases(text)
        + _find_private_keys(text)
        + _find_wallet_addresses(text)
        + _find_wallet_api_credentials(text)
    )

    sanitized_output = None
    if findings:
        sanitized_output = text
        for f in sorted(findings, key=lambda f: f["position"], reverse=True):
            start, end = f["position"], f.get("end_position", f["position"])
            sanitized_output = sanitized_output[:start] + f"[REDACTED:{f['type'].upper()}]" + sanitized_output[end:]

    for f in findings:
        f.pop("end_position", None)

    verdict = "NO_COMMIT" if findings else "COMMIT"
    risk_score = round(min(1.0, 0.6 + 0.1 * len(findings)), 3) if findings else 0.0
    confidence = round(1.0 - risk_score, 3)
    reason = (
        "; ".join(f"{f['type']} at offset {f['position']}" for f in findings)
        if findings else "No wallet secrets detected"
    )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "findings": findings,
        "sanitized_output": sanitized_output,
        "risk_score": risk_score,
    }


# ════════════════════════════════════════════════════════════════════════════════
# P1/P2 (standard) + X1-X4 (crypto-specific) — Prompt Firewall Crypto
# (dcl_evaluate_jailbreak_crypto)
#
# The tool's documented output schema restricts finding "type" to exactly five
# values: role_switch | instruction_override | token_smuggling |
# drain_wallet_injection | unlimited_approval_injection. X3 (trading-agent
# social engineering) is folded into instruction_override, and X4
# (framework-specific injection, e.g. fake "/buy" tool-call syntax) is folded
# into token_smuggling, since both are instructions smuggled in under a
# false-authority or false-format guise rather than a wholly new category.
# ════════════════════════════════════════════════════════════════════════════════
_ROLE_SWITCH_PATTERNS = [
    re.compile(r"(?i)\bpretend you are\b"),
    re.compile(r"(?i)\bact as if\b"),
    re.compile(r"(?i)\byou are now\b"),
    re.compile(r"(?i)\bignore your (?:previous|prior) (?:role|persona)\b"),
]

_INSTRUCTION_OVERRIDE_PATTERNS = [
    re.compile(r"(?i)\bignore (?:previous|prior|all prior|the above) instructions?\b"),
    re.compile(r"(?i)\bdisregard (?:the )?(?:above|previous instructions)\b"),
    # X3 — trading-agent social engineering
    re.compile(r"(?i)\bsecurity migration\b.{0,40}\b(?:re-?approve|re-?sign)\b"),
    re.compile(r"(?i)\bas the (?:exchange|protocol|platform) (?:operator|team)\b"),
    re.compile(r"(?i)\bact now or (?:lose|forfeit)\b"),
]

_TOKEN_SMUGGLING_PATTERNS = [
    re.compile(r"(?i)\bbase64\s*[:=]\s*[A-Za-z0-9+/=]{20,}"),
    re.compile(r"[\u200b\u200c\u200d\ufeff]"),  # zero-width characters
    # X4 — framework-specific injection (fake tool-call syntax)
    re.compile(r"(?m)^\s*/(?:buy|sell|approve|withdraw|transfer)\b"),
    re.compile(r"(?i)\b(?:Action|Tool)\s*:\s*(?:buy|sell|approve|withdraw|transfer)\b"),
]

_DRAIN_WALLET_PATTERNS = [
    re.compile(r"(?i)\btransfer all (?:funds|tokens|assets) to\b"),
    re.compile(r"(?i)\bsend (?:all|the entire|your entire) (?:balance|funds|tokens) to\b"),
    re.compile(r"(?i)\btest transaction\b.{0,40}\bfull balance\b"),
    re.compile(r"(?i)\bverification transfer\b.{0,40}\b(?:full|entire|whole) balance\b"),
    re.compile(r"(?i)\bwithdraw everything\b"),
]

_UNLIMITED_APPROVAL_PATTERNS = [
    re.compile(r"(?i)\btype\(uint256\)\.max\b"),
    re.compile(r"(?i)\bunlimited (?:allowance|approval)\b"),
    re.compile(r"(?i)\bapprove unlimited\b"),
    re.compile(r"(?i)\bskip (?:the )?slippage (?:check|confirmation)\b"),
    re.compile(r"(?i)\bbypass (?:the )?approval confirmation\b"),
]

_JAILBREAK_CRYPTO_GROUPS: List[Tuple[str, str, List[re.Pattern]]] = [
    ("role_switch", "major", _ROLE_SWITCH_PATTERNS),
    ("instruction_override", "major", _INSTRUCTION_OVERRIDE_PATTERNS),
    ("token_smuggling", "major", _TOKEN_SMUGGLING_PATTERNS),
    ("drain_wallet_injection", "critical", _DRAIN_WALLET_PATTERNS),
    ("unlimited_approval_injection", "critical", _UNLIMITED_APPROVAL_PATTERNS),
]

_JAILBREAK_CRYPTO_POLICY_VERSION = "crypto_jailbreak_v1"


def detect_jailbreak_crypto(text: str) -> dict:
    """Crypto-specialized instruction-override / jailbreak / injection detection,
    covering standard role-switch/instruction-override/token-smuggling patterns
    plus crypto-specific drain-wallet and unlimited-approval injection (X1/X2)."""
    findings = []
    for finding_type, severity, patterns in _JAILBREAK_CRYPTO_GROUPS:
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                findings.append({"type": finding_type, "severity": severity})
                break  # one finding per category is enough signal

    verdict = "NO_COMMIT" if findings else "COMMIT"
    has_critical = any(f["severity"] == "critical" for f in findings)
    confidence = 0.05 if has_critical else (0.4 if findings else 1.0)
    confidence = round(confidence, 3)
    reason = (
        "; ".join(f"{f['type']} ({f['severity']})" for f in findings)
        if findings else "No injection patterns matched"
    )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "findings": findings,
        "policy_version": _JAILBREAK_CRYPTO_POLICY_VERSION,
    }


# ════════════════════════════════════════════════════════════════════════════════
# M1-M4 — MEV Compliance (dcl_evaluate_mev)
#
# Text-level screen on what the agent says/proposes — not a mempool/transaction
# analyzer. regulatory_reference is an illustrative tag per finding type, not a
# legal citation.
# ════════════════════════════════════════════════════════════════════════════════
_FRONT_RUNNING_PATTERNS = [
    re.compile(r"(?i)\bfront[- ]?run(?:ning)?\b"),
    re.compile(r"(?i)\bback[- ]?run(?:ning)?\b"),
    re.compile(r"(?i)\bwatch(?:ing)? the mempool for\b"),
    re.compile(r"(?i)\binsert (?:a|my|our) transaction before\b"),
]
_SANDWICH_PATTERNS = [
    re.compile(r"(?i)\bsandwich(?:ing)?\s+(?:attack|their trade|the trade)\b"),
    re.compile(r"(?i)\bbuy before and sell after their trade\b"),
]
_WASH_TRADING_PATTERNS = [
    re.compile(r"(?i)\bwash trad(?:e|ing)\b"),
    re.compile(r"(?i)\btrade between (?:my|our) (?:own )?wallets\b"),
]
_LAYERING_PATTERNS = [
    re.compile(r"(?i)\blayer(?:ing)? (?:multiple )?orders?\b.{0,40}\b(?:manipulat|depth)\b"),
]
_SPOOFING_PATTERNS = [
    re.compile(r"(?i)\bspoof(?:ing)?\b"),
    re.compile(r"(?i)\bplace orders? with no intent to execute\b"),
]
_KYC_AML_PATTERNS = [
    re.compile(r"(?i)\b(?:mixer|tumbler)\b"),
    re.compile(r"(?i)\buntraceable transfer\b"),
    re.compile(r"(?i)\bstructure (?:the )?transfers? to stay under\b"),
    re.compile(r"(?i)\bobscure the (?:origin|destination) of funds\b"),
]
_PUMP_DUMP_PATTERNS = [
    re.compile(r"(?i)\bpump and dump\b"),
    re.compile(r"(?i)\bpump the price\b"),
    re.compile(r"(?i)\bexit liquidity\b"),
    re.compile(r"(?i)\bremove liquidity shortly after\b"),
    re.compile(r"(?i)\bcoordinated (?:buy|sell) timing\b"),
]

# (type, severity, regulatory_reference, patterns)
_MEV_GROUPS: List[Tuple[str, str, Optional[str], List[re.Pattern]]] = [
    ("front_running", "critical", "MiFID II", _FRONT_RUNNING_PATTERNS),
    ("sandwich_attack", "critical", "MiFID II", _SANDWICH_PATTERNS),
    ("wash_trading", "critical", "FCA", _WASH_TRADING_PATTERNS),
    ("layering", "critical", "FCA", _LAYERING_PATTERNS),
    ("spoofing", "critical", "FCA", _SPOOFING_PATTERNS),
    ("kyc_aml_flag", "critical", "EU AI Act Art. 13", _KYC_AML_PATTERNS),
    ("pump_and_dump", "major", "EU AI Act Art. 9", _PUMP_DUMP_PATTERNS),
]


def detect_mev(text: str) -> dict:
    """Screen for front-running/sandwich-attack patterns, wash trading/layering/
    spoofing, KYC/AML red flags, and pump-and-dump/rug-pull language (M1-M4)."""
    findings = []
    for finding_type, severity, reg_ref, patterns in _MEV_GROUPS:
        for pattern in patterns:
            if pattern.search(text):
                findings.append({"type": finding_type, "severity": severity, "regulatory_reference": reg_ref})
                break

    critical_count = sum(1 for f in findings if f["severity"] == "critical")
    major_count = sum(1 for f in findings if f["severity"] == "major")

    if critical_count > 0 or major_count >= 2:
        verdict = "NO_COMMIT"
        confidence = 0.05 if critical_count else 0.2
    elif major_count == 1:
        # Free-checklist WARN tier, collapsed conservatively to NO_COMMIT — see module docstring.
        verdict = "NO_COMMIT"
        confidence = 0.55
    else:
        verdict = "COMMIT"
        confidence = 1.0

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "findings": findings,
    }


# ════════════════════════════════════════════════════════════════════════════════
# V1-V4 — Trade Verifier (dcl_evaluate_trade)
# ════════════════════════════════════════════════════════════════════════════════
_GUARANTEED_RETURN_PATTERNS = [
    re.compile(r"(?i)\bguaranteed\s+(?:profit\b|return\b|\d+%)"),
    re.compile(r"(?i)\bcertain profit\b"),
    re.compile(r"(?i)\bguaranteed\s+\d+(?:\.\d+)?%"),
]
_ZERO_RISK_PATTERNS = [
    re.compile(r"(?i)\bno risk\b"),
    re.compile(r"(?i)\bcan'?t lose\b"),
    re.compile(r"(?i)\bcannot lose\b"),
    re.compile(r"(?i)\bzero (?:downside|risk)\b"),
    re.compile(r"(?i)\brisk[- ]free\b"),
]
_UNQUALIFIED_DIRECTIVE_PATTERNS = [
    re.compile(r"(?i)\b(?:buy|sell)\s+\S+\s+now\b"),
    re.compile(r"(?i)\b(?:buy|sell)\s+\S+\s+immediately\b"),
]

_TRADE_GROUPS: List[Tuple[str, str, List[re.Pattern]]] = [
    ("guaranteed_return", "critical", _GUARANTEED_RETURN_PATTERNS),
    ("zero_risk_claim", "critical", _ZERO_RISK_PATTERNS),
    ("unqualified_directive", "major", _UNQUALIFIED_DIRECTIVE_PATTERNS),
]


def detect_trade(text: str) -> dict:
    """Screen a trade decision's language for guaranteed-return claims, zero-risk
    framing, and unqualified buy/sell directives (V1-V3); require the word "risk"
    to appear somewhere for a clean COMMIT (V4)."""
    findings = []
    for finding_type, severity, patterns in _TRADE_GROUPS:
        for pattern in patterns:
            if pattern.search(text):
                findings.append({"type": finding_type, "severity": severity})
                break

    if findings:
        verdict = "NO_COMMIT"
        has_critical = any(f["severity"] == "critical" for f in findings)
        confidence = 0.05 if has_critical else 0.3
        reason = "; ".join(f"{f['type']} ({f['severity']})" for f in findings)
    elif "risk" in text.lower():
        verdict = "COMMIT"
        confidence = 1.0
        reason = "No unsafe trade language found; risk disclosure present"
    else:
        verdict = "NO_COMMIT"
        confidence = 0.5
        reason = "Missing required risk disclosure — no unsafe language, but the word 'risk' does not appear anywhere in the output"
        findings.append({"type": "missing_risk_disclosure", "severity": "major"})

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reason": reason,
        "findings": findings,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Semantic Drift Crypto (dcl_evaluate_signal)
#
# Pattern-based heuristic on the output text alone (no source document) — for a
# full claim-by-claim check against an actual price feed, the local grounding
# workflow (references/grounding-workflow.md) is the right tool instead.
# ════════════════════════════════════════════════════════════════════════════════
_GUARANTEED_PRICE_PATTERNS = [
    re.compile(r"(?i)\bwill definitely (?:reach|hit)\b"),
    re.compile(r"(?i)\bguaranteed to (?:reach|hit)\b"),
    re.compile(r"(?i)\bwill surely (?:reach|hit|moon)\b"),
    re.compile(r"(?i)\bcertain to (?:reach|hit)\b"),
]
_ABSOLUTE_CERTAINTY_PATTERNS = [
    re.compile(r"(?i)\b100%\s*certain\b"),
    re.compile(r"(?i)\bwithout any doubt\b"),
    re.compile(r"(?i)\bcannot go down\b"),
    re.compile(r"(?i)\bcannot fail\b"),
    re.compile(r"(?i)\bguaranteed profit\b"),
]
_DOLLAR_FIGURE = re.compile(r"\$\s?[0-9][0-9,]*(?:\.[0-9]+)?")

_KNOWN_TICKERS = frozenset({
    "BTC", "ETH", "SOL", "USDC", "USDT", "BNB", "XRP", "ADA", "DOGE", "MATIC",
    "AVAX", "LINK", "DOT", "LTC", "TRX", "SHIB", "UNI", "ATOM", "XLM", "NEAR",
    "BASE", "ARB", "OP", "APT", "SUI", "TON", "ETC", "BCH", "FIL", "ICP",
})
_CASHTAG = re.compile(r"\$([A-Z]{2,6})\b")


def detect_signal(text: str) -> dict:
    """Flag fabricated prices, invented token names, contradicted indicators, and
    overconfident/certain-outcome language in market-signal output."""
    findings = []
    guaranteed_hit = False

    for pattern in _GUARANTEED_PRICE_PATTERNS:
        m = pattern.search(text)
        if m:
            findings.append({"type": "guaranteed_price_prediction", "severity": "critical"})
            guaranteed_hit = True
            break

    for pattern in _ABSOLUTE_CERTAINTY_PATTERNS:
        if pattern.search(text):
            findings.append({"type": "absolute_certainty_claim", "severity": "critical"})
            break

    # fabricated_price: a specific $ figure appearing in the same breath as a
    # guaranteed/certain-outcome claim — pattern-only, so treated as "major"
    # rather than "critical" (no ground-truth price feed to confirm against).
    if guaranteed_hit and _DOLLAR_FIGURE.search(text):
        findings.append({"type": "fabricated_price", "severity": "major"})

    for m in _CASHTAG.finditer(text):
        ticker = m.group(1)
        if ticker not in _KNOWN_TICKERS:
            findings.append({"type": "invented_token", "severity": "major"})
            break  # one flag is enough signal per call

    critical_count = sum(1 for f in findings if f["severity"] == "critical")
    major_count = sum(1 for f in findings if f["severity"] == "major")

    if critical_count > 0 or major_count >= 2:
        verdict = "NO_COMMIT"
        confidence = 0.05 if critical_count else 0.2
    elif major_count == 1:
        verdict = "NO_COMMIT"
        confidence = 0.55
    else:
        verdict = "COMMIT"
        confidence = 1.0

    reason = (
        "; ".join(f"{f['type']} ({f['severity']})" for f in findings)
        if findings else "No fabricated-price or overconfidence patterns matched"
    )

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reason": reason,
        "findings": findings,
    }

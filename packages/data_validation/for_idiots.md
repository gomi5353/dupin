# RDF/OWL Dataset Validation 入門

這份文件是給第一次接觸 RDF、OWL、Turtle 和 SHACL 的讀者。目標是理解
`src/data_validation/ontology/` 裡的三個 `.ttl` 檔案，並知道它們如何協助檢查
LeRobot Dataset v3。

## 第一節：每個檔案負責什麼？

這三個檔案不是三份重複的設定，而是三個不同層次：

| 檔案 | 負責內容 | 白話說法 |
|---|---|---|
| `lerobot.ttl` | OWL ontology | 定義有哪些概念、哪些關係，以及每個關係的來源與目標類型 |
| `lerobot-shapes.ttl` | SHACL shapes | 定義驗證規則，例如 dataset 必須是 v3、必須有 episode、每個 frame 必須有 action |
| `hcis-capstone.ttl` | 本 repo 的資料契約 | 定義本 repo 要求的 feature 名稱、dtype 和 shape |

可以把它們想成：

```text
lerobot.ttl
    定義「名詞」和「文法」
        ↓
lerobot-shapes.ttl
    定義「必須符合的規則」
        ↓
hcis-capstone.ttl
    定義「本課程的實際欄位規格」
```

### `lerobot.ttl`：Ontology

這是最基礎的世界模型。它說明：

- `Dataset` 是 dataset
- `Episode` 是 episode
- `Frame` 是一個時間 frame
- `Feature` 是 state、action 或 image 等資料欄位
- `hasEpisode` 表示 dataset 擁有 episode
- `hasFrame` 表示 episode 擁有 frame
- `hasAction` 表示 frame 有 action

它本身不會說「dataset 一定要有五個 feature」。它只描述概念與關係。

### `lerobot-shapes.ttl`：SHACL 規則

這是檢查規則。例如：

- Dataset 的版本必須是 `v3.0`
- Dataset 必須有至少一個 episode
- Dataset 必須有至少一個 task
- Episode 必須有 frame
- Frame 必須有 state、action、front image 和 wrist image
- `frameIndex` 必須是大於等於 0 的整數

SHACL 很適合做「缺少欄位就報錯」的 validation。OWL 比較像資料模型，SHACL
才是這裡實際執行的檢查規則。

### `hcis-capstone.ttl`：本 repo 的 contract

這份檔案描述本專案期待的 feature：

| RDF resource | LeRobot feature | dtype | shape |
|---|---|---|---|
| `lr:ObservationState` | `observation.state` | `float32` | `[9]` |
| `lr:Action` | `action` | `float32` | `[8]` |
| `lr:FrontCamera` | `observation.images.front` | `video` | `[480,640,3]` |
| `lr:WristCamera` | `observation.images.wrist` | `video` | `[480,640,3]` |
| `lr:TaskIndex` | `task_index` | `int64` | `[1]` |

LeRobot v3 通常會把 task vocabulary 放在 `meta/tasks.parquet`，再用
`task_index` 指向 task。因此這裡使用 `task_index`，不是強迫 dataset 一定要
有名為 `task` 的 tensor feature。

## 第二節：逐行解釋 `.ttl` 檔案

### A. `lerobot.ttl`

#### 第 1–4 行：縮寫名稱

```turtle
@prefix lr: <https://hcis.example.org/lerobot#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
```

- 第 1 行：把很長的 ontology namespace 縮寫成 `lr:`。所以 `lr:Dataset`
  實際上代表 `https://hcis.example.org/lerobot#Dataset`。
- 第 2 行：引入 OWL vocabulary，例如 `owl:Class` 和
  `owl:ObjectProperty`。
- 第 3 行：引入 RDFS vocabulary，例如 `rdfs:domain`、`rdfs:range`、
  `rdfs:subClassOf` 和 `rdfs:label`。
- 第 4 行：引入 XML Schema datatype，例如 `xsd:string` 和
  `xsd:integer`。
- 每行最後的 `.` 表示這個 Turtle statement 結束。

#### 第 6–7 行：ontology 本身

```turtle
lr:LeRobotDatasetOntology a owl:Ontology ;
    rdfs:label "LeRobot Dataset ontology"@en .
```

- `lr:LeRobotDatasetOntology` 是 ontology 的名稱。
- `a owl:Ontology` 表示它是一個 OWL ontology；`a` 是 `rdf:type` 的縮寫。
- `;` 表示同一個 subject 還有下一個 predicate。
- `rdfs:label` 提供人類可讀的名稱。
- `@en` 表示 label 是英文文字。

#### 第 9–18 行：定義 classes

```turtle
lr:Dataset a owl:Class .
lr:DatasetContract a owl:Class .
lr:Episode a owl:Class .
lr:Frame a owl:Class .
lr:Feature a owl:Class .
lr:StateFeature a owl:Class ; rdfs:subClassOf lr:Feature .
lr:ActionFeature a owl:Class ; rdfs:subClassOf lr:Feature .
lr:ImageFeature a owl:Class ; rdfs:subClassOf lr:Feature .
lr:Task a owl:Class .
lr:Robot a owl:Class .
```

- 每個 `a owl:Class` 都是在宣告一個概念類別。
- `lr:Dataset`：整個 LeRobot dataset。
- `lr:DatasetContract`：資料契約，不是實際收集到的 dataset。
- `lr:Episode`：一個 demonstration episode。
- `lr:Frame`：episode 中的一個時間點。
- `lr:Feature`：dataset 的資料欄位。
- `lr:StateFeature`：state feature，而且是一般 `Feature` 的子類別。
- `lr:ActionFeature`：action feature，而且是一般 `Feature` 的子類別。
- `lr:ImageFeature`：image/video feature，而且是一般 `Feature` 的子類別。
- `lr:Task`：任務描述或 task vocabulary 中的一個 task。
- `lr:Robot`：機器人種類，例如 Franka Panda。
- `rdfs:subClassOf` 表示繼承關係。因為 `StateFeature` 是 `Feature` 的子類別，
  state 也可以被當成一般 feature 使用。

#### 第 20–30 行：定義 object properties

```turtle
lr:hasEpisode a owl:ObjectProperty ; rdfs:domain lr:Dataset ; rdfs:range lr:Episode .
```

- `lr:hasEpisode` 是一個 object property，表示 subject 和另一個 RDF resource
  之間的關係。
- `rdfs:domain lr:Dataset` 表示關係的 subject 應該是 Dataset。
- `rdfs:range lr:Episode` 表示關係的 object 應該是 Episode。
- 白話就是：`dataset hasEpisode episode`。

其餘各行使用完全相同的語法：

```turtle
lr:hasFrame a owl:ObjectProperty ; rdfs:domain lr:Episode ; rdfs:range lr:Frame .
```

Episode 可以擁有 Frame。

```turtle
lr:hasFeature a owl:ObjectProperty ; rdfs:domain lr:Dataset ; rdfs:range lr:Feature .
```

Dataset 可以擁有 Feature。

```turtle
lr:contractFeature a owl:ObjectProperty ; rdfs:domain lr:DatasetContract ; rdfs:range lr:Feature .
```

Dataset contract 可以列出它要求的 Feature。這和 `hasFeature` 不同：
`hasFeature` 是實際 dataset 的欄位，`contractFeature` 是規格要求的欄位。

```turtle
lr:hasTask a owl:ObjectProperty ; rdfs:domain lr:Dataset ; rdfs:range lr:Task .
```

Dataset 可以擁有 task。

```turtle
lr:usesRobot a owl:ObjectProperty ; rdfs:domain lr:Dataset ; rdfs:range lr:Robot .
```

Dataset 會使用某一種 robot。

```turtle
lr:hasObservationState a owl:ObjectProperty ; rdfs:domain lr:Frame ; rdfs:range lr:StateFeature .
lr:hasAction a owl:ObjectProperty ; rdfs:domain lr:Frame ; rdfs:range lr:ActionFeature .
lr:hasFrontImage a owl:ObjectProperty ; rdfs:domain lr:Frame ; rdfs:range lr:ImageFeature .
lr:hasWristImage a owl:ObjectProperty ; rdfs:domain lr:Frame ; rdfs:range lr:ImageFeature .
```

這四行描述一個 Frame 裡面應該有哪些語意資料：state、action、front image
和 wrist image。它們的 range 也限制了資料類型。

```turtle
lr:belongsToEpisode a owl:ObjectProperty ; rdfs:domain lr:Frame ; rdfs:range lr:Episode .
```

這表示一個 Frame 屬於某個 Episode。這個反向關係讓 validator 可以確認 frame
不是孤兒資料。

#### 第 32–38 行：定義 datatype properties

```turtle
lr:formatVersion a owl:DatatypeProperty ; rdfs:domain lr:Dataset ; rdfs:range xsd:string .
```

Dataset 的格式版本是文字，例如 `"v3.0"`。

```turtle
lr:datasetFPS a owl:DatatypeProperty ; rdfs:domain lr:Dataset ; rdfs:range xsd:integer .
```

Dataset 的 FPS 必須是整數，例如 `30` 或 `60`。

```turtle
lr:featureName a owl:DatatypeProperty ; rdfs:domain lr:Feature ; rdfs:range xsd:string .
```

Feature 的實際名稱，例如 `"action"` 或
`"observation.images.front"`。

```turtle
lr:featureDType a owl:DatatypeProperty ; rdfs:domain lr:Feature ; rdfs:range xsd:string .
```

Feature 的 dtype，例如 `"float32"`、`"int64"` 或 `"video"`。

```turtle
lr:featureShape a owl:DatatypeProperty ; rdfs:domain lr:Feature ; rdfs:range xsd:string .
```

Feature 的 shape。這裡使用字串表示，例如 `"[9]"` 或 `"[480,640,3]"`，
因為 shape 是一個多維陣列，不是單一 XML datatype。

```turtle
lr:frameIndex a owl:DatatypeProperty ; rdfs:domain lr:Frame ; rdfs:range xsd:integer .
```

Frame 的時間索引，例如 0、1、2、3。

```turtle
lr:episodeOutcome a owl:DatatypeProperty ; rdfs:domain lr:Episode ; rdfs:range xsd:string .
```

Episode 的結果，例如 `"success"`、`"failure"` 或 `"aborted"`。

### B. `lerobot-shapes.ttl`

#### 第 1–3 行：引入 namespace

```turtle
@prefix lr: <https://hcis.example.org/lerobot#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
```

- `lr:` 指向我們自己的 ontology。
- `sh:` 引入 SHACL vocabulary，例如 `sh:NodeShape`、`sh:property` 和
  `sh:minCount`。
- `xsd:` 用來指定資料型別。

#### 第 5–11 行：DatasetShape

```turtle
lr:DatasetShape a sh:NodeShape ;
    sh:targetClass lr:Dataset ;
```

- `lr:DatasetShape` 是一組 validation rules。
- `a sh:NodeShape` 表示它是 SHACL node shape。
- `sh:targetClass lr:Dataset` 表示所有 RDF 中被標記成 `lr:Dataset` 的 node
  都要套用這些規則。

```turtle
    sh:property [ sh:path lr:formatVersion ; sh:minCount 1 ; sh:hasValue "v3.0" ] ;
```

- `sh:property [ ... ]` 建立一個匿名 property shape。
- `sh:path lr:formatVersion` 指定要檢查的欄位。
- `sh:minCount 1` 表示至少要有一個值。
- `sh:hasValue "v3.0"` 表示值必須正好是 `v3.0`。

```turtle
    sh:property [ sh:path lr:datasetFPS ; sh:minCount 1 ; sh:datatype xsd:integer ; sh:minInclusive 1 ] ;
```

Dataset 必須有 FPS；FPS 必須是整數，而且至少為 1。

```turtle
    sh:property [ sh:path lr:hasEpisode ; sh:minCount 1 ; sh:class lr:Episode ] ;
```

Dataset 至少要有一個 episode，而且 object 必須是 `lr:Episode`。

```turtle
    sh:property [ sh:path lr:hasTask ; sh:minCount 1 ; sh:class lr:Task ] ;
```

Dataset 至少要有一個 task，而且 task resource 必須屬於 `lr:Task`。

```turtle
    sh:property [ sh:path lr:hasFeature ; sh:minCount 5 ; sh:class lr:Feature ] .
```

Dataset 至少要有五個 feature，而且每個 feature 必須屬於 `lr:Feature`。
最後的 `.` 結束整個 `DatasetShape`。

#### 第 13–15 行：EpisodeShape

```turtle
lr:EpisodeShape a sh:NodeShape ;
    sh:targetClass lr:Episode ;
    sh:property [ sh:path lr:hasFrame ; sh:minCount 1 ; sh:class lr:Frame ] .
```

- 所有 `lr:Episode` 都要套用 `EpisodeShape`。
- 每個 episode 至少要有一個 frame。
- frame 必須是 `lr:Frame`。

#### 第 17–24 行：FrameShape

```turtle
lr:FrameShape a sh:NodeShape ;
    sh:targetClass lr:Frame ;
```

所有 `lr:Frame` 都要套用下面的 frame rules。

```turtle
    sh:property [ sh:path lr:frameIndex ; sh:minCount 1 ; sh:datatype xsd:integer ; sh:minInclusive 0 ] ;
```

每個 frame 必須有 `frameIndex`；它必須是大於等於 0 的整數。

```turtle
    sh:property [ sh:path lr:belongsToEpisode ; sh:minCount 1 ; sh:maxCount 1 ; sh:class lr:Episode ] ;
```

每個 frame 必須屬於剛好一個 episode：

- `sh:minCount 1`：不能沒有 episode
- `sh:maxCount 1`：不能同時屬於多個 episode
- `sh:class lr:Episode`：episode 必須是正確類別

```turtle
    sh:property [ sh:path lr:hasObservationState ; sh:minCount 1 ; sh:maxCount 1 ; sh:class lr:StateFeature ] ;
```

每個 frame 必須有剛好一個 state feature，而且它必須是 `StateFeature`。

```turtle
    sh:property [ sh:path lr:hasAction ; sh:minCount 1 ; sh:maxCount 1 ; sh:class lr:ActionFeature ] ;
```

每個 frame 必須有剛好一個 action feature，而且它必須是 `ActionFeature`。

```turtle
    sh:property [ sh:path lr:hasFrontImage ; sh:minCount 1 ; sh:maxCount 1 ; sh:class lr:ImageFeature ] ;
```

每個 frame 必須有剛好一個 front image feature，而且它必須是 `ImageFeature`。

```turtle
    sh:property [ sh:path lr:hasWristImage ; sh:minCount 1 ; sh:maxCount 1 ; sh:class lr:ImageFeature ] .
```

每個 frame 必須有剛好一個 wrist image feature，而且它必須是 `ImageFeature`。

### C. `hcis-capstone.ttl`

#### 第 1 行：namespace

```turtle
@prefix lr: <https://hcis.example.org/lerobot#> .
```

使用前面 `lerobot.ttl` 定義的 `lr:` namespace。這份檔案不需要重新引入
OWL、RDFS 或 XSD，因為它只是在使用已定義的 resource。

#### 第 3–4 行：宣告資料契約

```turtle
lr:HCISDatasetContract a lr:DatasetContract ;
    lr:contractFeature lr:ObservationState, lr:Action, lr:FrontCamera, lr:WristCamera, lr:TaskIndex .
```

- `lr:HCISDatasetContract` 是本 repo 的資料契約。
- `a lr:DatasetContract` 表示它是 contract，不是實際 dataset。
- `lr:contractFeature` 列出 contract 要求的五個 feature。
- 逗號表示同一個 predicate 有多個 object。
- 這裡的 `lr:TaskIndex` 對應 LeRobot v3 的 `task_index`。

#### 第 6–7 行：state feature

```turtle
lr:ObservationState a lr:StateFeature ;
    lr:featureName "observation.state" ; lr:featureDType "float32" ; lr:featureShape "[9]" .
```

這個 resource 是一個 state feature，實際欄位名稱是
`observation.state`，dtype 是 `float32`，shape 是 `[9]`。

#### 第 8–9 行：action feature

```turtle
lr:Action a lr:ActionFeature ;
    lr:featureName "action" ; lr:featureDType "float32" ; lr:featureShape "[8]" .
```

這個 resource 是 action feature，欄位名稱是 `action`，dtype 是 `float32`，
shape 是 `[8]`。

#### 第 10–11 行：front camera

```turtle
lr:FrontCamera a lr:ImageFeature ;
    lr:featureName "observation.images.front" ; lr:featureDType "video" ; lr:featureShape "[480,640,3]" .
```

這是 front camera image feature：高度 480、寬度 640、RGB 三個 channel。
`video` 表示它在 LeRobot metadata 中是影像／影片類型。

#### 第 12–13 行：wrist camera

```turtle
lr:WristCamera a lr:ImageFeature ;
    lr:featureName "observation.images.wrist" ; lr:featureDType "video" ; lr:featureShape "[480,640,3]" .
```

這和 front camera 相同，但欄位名稱是 `observation.images.wrist`，代表手腕相機。

#### 第 14 行：task index

```turtle
lr:TaskIndex a lr:Feature ; lr:featureName "task_index" ; lr:featureDType "int64" ; lr:featureShape "[1]" .
```

這個 feature 儲存 task vocabulary 的 index：

- feature name：`task_index`
- dtype：`int64`
- shape：`[1]`

實際 task 名稱則從 `meta/tasks.parquet` 取得。

## 執行驗證

```bash
uv run python scripts/validate_dataset.py \
    --dataset ./datasets/<dataset-name> \
    --contract packages/data_validation/src/data_validation/ontology/hcis-capstone.ttl \
    --strict
```

validator 會先檢查實際檔案，再把 metadata 轉為 RDF，最後載入
`lerobot.ttl`、`lerobot-shapes.ttl` 和指定的 contract 進行驗證。

成功時表示 dataset 同時通過：

- LeRobot v3 metadata
- parquet columns 與 frame count
- video/image decode
- HCIS feature contract
- RDF/OWL model
- SHACL constraints

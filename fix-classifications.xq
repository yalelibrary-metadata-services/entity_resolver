(: count(distinct-values(//*[@name = "personId"])) :)
(: distinct-values(db:get("training_dataset")//*[@name = "personId"])[not(. = distinct-values(db:get("parallel_classifications")/fn:map/fn:map/@key))] :)

(: for $m in /fn:map/fn:map
group by $k := $m/fn:array[@key = "label"]/fn:string[1]
order by count($m) descending
return <_ k="{$k}" c="{count($m)}"/> :)

for $m in /fn:map/fn:map
let $r := $m/fn:string[@key = "rationale"]
where $r[text() contains text "veteran"]
where $m[fn:array[@key = "label"][1][not(fn:string = "Military, Security, and Defense")]]
return (
  delete node $m/fn:array,
  insert node (
    <fn:array key="label">
      <fn:string>Military, Security, and Defense</fn:string>
      <fn:string>Politics, Policy, and Government</fn:string>
    </fn:array>
    ,
    <fn:array key="path">
      <fn:string>Society, Governance, and Public Life > Military, Security, and Defense</fn:string>
      <fn:string>Society, Governance, and Public Life > Politics, Policy, and Government</fn:string>
    </fn:array>
  ) as first into $m
)

(:

"label": [
      "Military, Security, and Defense",
      "Politics, Policy, and Government"
    ],
    "path": [
      "Society, Governance, and Public Life > Military, Security, and Defense",
      "Society, Governance, and Public Life > Politics, Policy, and Government"
    ]

:)
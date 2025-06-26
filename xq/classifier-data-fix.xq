let $c := "parallel_classifications"
let $t := "training_dataset_classified_2025-06-25"
let $recs :=
  for $rec in db:get($t)/csv/record
  let $id := $rec/*[@name = "personId"]
  where not(db:attribute($c, $id))
  (: 
  let $_ := substring-after($id, "#")
  let $recordId := data($rec/*[@name = "recordId"])
  group by $k := $_ || $recordId
  where count($rec) gt 1 :)
  (: let $text := string-join($rec/*)
  group by $k := $text
  where count($rec) gt 1 :)
  return $rec

(: for $rec at $p in $recs, $match in db:get($c)/fn:map/fn:map
where substring-before($match/@key, "-") = $rec/*[@name = "recordId"]
group by $k := $rec/*[@name = "personId"] :)



(: <_>{$rec[1]/*[@name = "person"], $rec[1]/*[@name = "personId"], $match}</_> :)
return <csv>{$recs}</csv> => csv:serialize({"header": true(), "format": "attributes"})



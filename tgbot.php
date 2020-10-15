<?php

    $token = '<BOT ID>';
    $j = json_decode(file_get_contents("php://input"));

    function appendFileUnique($fp, $line)
    {
        $data = file_get_contents($fp);
        if(strpos($data, $line . "\n") === false)
            file_put_contents($fp, $line . "\n", FILE_APPEND | LOCK_EX);
    }

    if(isset($j->{'message'}->{'text'}))
    {
        if(strpos($j->{'message'}->{'text'}, "/quote") !== FALSE)
        {
            $file = file("out.txt"); 
            $line = $file[rand(0, count($file) - 1)];
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode($line));
            http_response_code(200);
            exit;
        }
        
        if(strpos($j->{'message'}->{'text'}, "/info") !== FALSE)
        {
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode(file_get_contents("portstat.txt")));
            http_response_code(200);
            exit;
        }

        $msg = $j->{'message'}->{'text'};

        $pp = explode(' ', $msg);
        $pps = array_slice($pp, 0, 16);

        $str = "";
        foreach($pps as $p)
            if(strlen($p) <= 250 && $p != "" && $p != " ")
                $str .= str_replace("\n", "", $p) . " ";
        $str = rtrim($str, ' ');

        appendFileUnique("botmsg.txt", str_replace("\n", "", substr($str, 0, 4090))); //you could reduce this to 768 or 1024 safely

        foreach($pp as $p)
            if(strlen($p) <= 250 && $p != "" && $p != " ")
                appendFileUnique("botdict.txt", str_replace("\n", "", substr($p, 0, 250)));
    }

    http_response_code(200);

?>
